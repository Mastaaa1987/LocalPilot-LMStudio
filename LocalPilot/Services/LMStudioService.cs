using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LocalPilot.Services
{
    /// <summary>
    /// Communicates with LM Studio through its OpenAI-compatible local API.
    /// Supports model discovery, chat/completion streaming, tool calls and embeddings.
    /// </summary>
    public class LMStudioService
    {
        private static readonly HttpClient _httpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        private static readonly HttpClient _backgroundHttpClient = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        private string _baseUrl;
        private volatile bool _circuitBreakerTripped;
        private int _consecutiveFailures;
        private DateTime _circuitBreakerCooldownUntil = DateTime.MinValue;

        public bool CircuitBreakerTripped
        {
            get => _circuitBreakerTripped;
            private set => _circuitBreakerTripped = value;
        }

        public LMStudioService(string baseUrl = "http://localhost:1234/v1")
        {
            UpdateBaseUrl(baseUrl);
        }

        public void UpdateBaseUrl(string baseUrl)
        {
            var normalized = string.IsNullOrWhiteSpace(baseUrl) ? "http://localhost:1234/v1" : baseUrl.Trim().TrimEnd('/');
            _baseUrl = normalized.EndsWith("/v1", StringComparison.OrdinalIgnoreCase) ? normalized : normalized + "/v1";
        }

        private HttpRequestMessage CreateJsonRequest(string path, object payload)
        {
            return new HttpRequestMessage(HttpMethod.Post, _baseUrl + path)
            {
                Content = new StringContent(JsonConvert.SerializeObject(payload, new JsonSerializerSettings
                {
                    NullValueHandling = NullValueHandling.Ignore
                }), Encoding.UTF8, "application/json")
            };
        }

        private static List<object> ToOpenAiMessages(IEnumerable<ChatMessage> messages)
        {
            return (messages ?? Enumerable.Empty<ChatMessage>())
                .Select(m => (object)new
                {
                    // LocalPilot historically stored tool results without tool_call_id.
                    // Sending these as user context is valid OpenAI-format input and keeps agent loops compatible.
                    role = string.Equals(m.Role, "tool", StringComparison.OrdinalIgnoreCase) ? "user" : m.Role,
                    content = m.Content
                }).ToList();
        }

        public async Task WarmupAsync(string model, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(model) || CircuitBreakerTripped) return;
            try
            {
                var payload = new { model, messages = new[] { new { role = "user", content = "Hi" } }, max_tokens = 1, stream = false };
                using (var request = CreateJsonRequest("/chat/completions", payload))
                using (var localCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    localCts.CancelAfter(TimeSpan.FromMinutes(3));
                    using (var response = await _httpClient.SendAsync(request, localCts.Token).ConfigureAwait(false))
                        response.EnsureSuccessStatusCode();
                }
                LocalPilotLogger.Log($"LM Studio warmup complete for model: {model}", LogCategory.LMStudio);
            }
            catch { /* Best effort only. */ }
        }

        public async Task<List<string>> GetAvailableModelsAsync(CancellationToken ct = default)
        {
            var names = new List<string>();
            try
            {
                using (var localCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    localCts.CancelAfter(TimeSpan.FromSeconds(10));
                    using (var response = await _httpClient.GetAsync(_baseUrl + "/models", localCts.Token).ConfigureAwait(false))
                    {
                        if (!response.IsSuccessStatusCode) return names;
                        var obj = JObject.Parse(await response.Content.ReadAsStringAsync().ConfigureAwait(false));
                        foreach (var model in obj["data"] as JArray ?? new JArray())
                        {
                            var id = model["id"]?.ToString();
                            if (!string.IsNullOrWhiteSpace(id)) names.Add(id);
                        }
                    }
                }
                ResetCircuitBreaker();
            }
            catch { /* LM Studio server not running. */ }
            return names;
        }

        public async Task<bool> IsAvailableAsync(CancellationToken ct = default)
        {
            try
            {
                using (var localCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    localCts.CancelAfter(TimeSpan.FromSeconds(5));
                    using (var response = await _httpClient.GetAsync(_baseUrl + "/models", localCts.Token).ConfigureAwait(false))
                    {
                        if (response.IsSuccessStatusCode) ResetCircuitBreaker();
                        return response.IsSuccessStatusCode;
                    }
                }
            }
            catch { return false; }
        }

        private static string ComputeHash(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            using (var sha = System.Security.Cryptography.SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(text))).Replace("-", "").ToLowerInvariant();
        }

        public async Task<float[]> GetEmbeddingsAsync(string model, string prompt, CancellationToken ct = default)
        {
            var results = await GetEmbeddingsBatchAsync(model, new List<string> { prompt }, ct).ConfigureAwait(false);
            return results.Count > 0 ? results[0] : null;
        }

        public async Task<List<float[]>> GetEmbeddingsBatchAsync(string model, List<string> prompts, CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(model) || prompts == null || prompts.Count == 0) return new List<float[]>();
            if (CircuitBreakerTripped && DateTime.Now < _circuitBreakerCooldownUntil)
                return prompts.Select(_ => (float[])null).ToList();
            if (CircuitBreakerTripped) ResetCircuitBreaker();

            var results = new List<float[]>(new float[prompts.Count][]);
            var missingIndices = new List<int>();
            var missingPrompts = new List<string>();
            for (var i = 0; i < prompts.Count; i++)
            {
                if (string.IsNullOrWhiteSpace(prompts[i])) continue;
                var cached = await StorageService.Instance.GetCachedEmbeddingAsync($"{model}:{ComputeHash(prompts[i])}");
                if (cached != null) results[i] = cached;
                else { missingIndices.Add(i); missingPrompts.Add(prompts[i]); }
            }
            if (missingIndices.Count == 0) return results;

            try
            {
                using (var request = CreateJsonRequest("/embeddings", new { model, input = missingPrompts }))
                using (var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    requestCts.CancelAfter(TimeSpan.FromMinutes(5));
                    using (var response = await _backgroundHttpClient.SendAsync(request, requestCts.Token).ConfigureAwait(false))
                    {
                        response.EnsureSuccessStatusCode();
                        var data = JObject.Parse(await response.Content.ReadAsStringAsync().ConfigureAwait(false))["data"] as JArray;
                        if (data == null) return results;
                        foreach (var item in data.OrderBy(x => x["index"]?.Value<int>() ?? 0))
                        {
                            var batchIndex = item["index"]?.Value<int>() ?? 0;
                            if (batchIndex < 0 || batchIndex >= missingIndices.Count) continue;
                            var vector = item["embedding"]?.ToObject<float[]>();
                            results[missingIndices[batchIndex]] = vector;
                            if (vector != null)
                                await StorageService.Instance.StoreCachedEmbeddingAsync($"{model}:{ComputeHash(missingPrompts[batchIndex])}", vector);
                        }
                    }
                }
                ResetCircuitBreaker();
            }
            catch (Exception ex)
            {
                LocalPilotLogger.LogError("[LM Studio] Embedding request failed", ex, LogCategory.LMStudio);
                HandleFailure();
            }
            return results;
        }

        private void HandleFailure()
        {
            if (Interlocked.Increment(ref _consecutiveFailures) < 5) return;
            CircuitBreakerTripped = true;
            _circuitBreakerCooldownUntil = DateTime.Now.AddMinutes(5);
            LocalPilotLogger.Log("LM Studio circuit breaker tripped; cooldown active for 5 minutes.", LogCategory.LMStudio, LogSeverity.Warning);
        }

        private void ResetCircuitBreaker()
        {
            Interlocked.Exchange(ref _consecutiveFailures, 0);
            CircuitBreakerTripped = false;
        }

        public async IAsyncEnumerable<string> StreamCompletionAsync(
            string model,
            string prompt,
            LMStudioOptions options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            var messages = new List<ChatMessage> { new ChatMessage { Role = "user", Content = prompt } };
            await foreach (var result in StreamChatWithToolsAsync(model, messages, null, options, ct))
                if (result.IsTextToken) yield return result.TextToken;
        }

        public async IAsyncEnumerable<string> StreamChatAsync(
            string model,
            List<ChatMessage> messages,
            LMStudioOptions options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await foreach (var result in StreamChatWithToolsAsync(model, messages, null, options, ct))
                if (result.IsTextToken) yield return result.TextToken;
        }

        public async IAsyncEnumerable<ChatStreamResult> StreamChatWithToolsAsync(
            string model,
            List<ChatMessage> messages,
            List<LMStudioToolDefinition> tools = null,
            LMStudioOptions options = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            if (CircuitBreakerTripped)
            {
                yield return ChatStreamResult.Text("\n[LocalPilot] LM Studio is currently unreachable. Ensure the local server is running at " + _baseUrl);
                yield break;
            }

            options = options ?? new LMStudioOptions();
            using (var requestCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
            {
                requestCts.CancelAfter(TimeSpan.FromSeconds(options.RequestTimeoutSeconds > 0 ? options.RequestTimeoutSeconds : 120));
                var payload = new
                {
                    model,
                    messages = ToOpenAiMessages(messages),
                    tools = tools != null && tools.Count > 0 ? tools : null,
                    stream = true,
                    temperature = options.Temperature,
                    top_p = options.TopP,
                    max_tokens = options.NumPredict,
                    stop = options.Stop != null && options.Stop.Count > 0 ? options.Stop : null
                };

                HttpResponseMessage response = null;
                bool usedToolFallback = false;
                string requestError = null;
                bool wasCancelled = false;
                try
                {
                    using (var request = CreateJsonRequest("/chat/completions", payload))
                        response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, requestCts.Token).ConfigureAwait(false);

                    if (response.StatusCode == HttpStatusCode.BadRequest && tools != null && tools.Count > 0)
                    {
                        response.Dispose();
                        var fallback = new
                        {
                            model,
                            messages = ToOpenAiMessages(messages),
                            stream = true,
                            temperature = options.Temperature,
                            top_p = options.TopP,
                            max_tokens = options.NumPredict,
                            stop = options.Stop != null && options.Stop.Count > 0 ? options.Stop : null
                        };
                        using (var fallbackRequest = CreateJsonRequest("/chat/completions", fallback))
                            response = await _httpClient.SendAsync(fallbackRequest, HttpCompletionOption.ResponseHeadersRead, requestCts.Token).ConfigureAwait(false);
                        usedToolFallback = true;
                    }
                    response.EnsureSuccessStatusCode();
                    ResetCircuitBreaker();
                }
                catch (OperationCanceledException)
                {
                    response?.Dispose();
                    wasCancelled = true;
                }
                catch (Exception ex)
                {
                    response?.Dispose();
                    HandleFailure();
                    requestError = ex.Message;
                }

                if (wasCancelled)
                {
                    if (!ct.IsCancellationRequested)
                        yield return ChatStreamResult.Text($"\n[LocalPilot] LM Studio request timed out after {options.RequestTimeoutSeconds} seconds.");
                    yield break;
                }
                if (requestError != null)
                {
                    yield return ChatStreamResult.Text($"\n[LocalPilot] Could not reach LM Studio: {requestError}");
                    yield break;
                }
                if (usedToolFallback)
                    yield return ChatStreamResult.Text("\n> The loaded LM Studio model does not support tool calling; continuing without native tools.\n\n");

                var pendingCalls = new Dictionary<int, PendingToolCall>();
                using (response)
                using (var stream = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
                using (var reader = new StreamReader(stream))
                {
                    while (!reader.EndOfStream && !requestCts.IsCancellationRequested)
                    {
                        var line = await reader.ReadLineAsync().ConfigureAwait(false);
                        if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data:", StringComparison.OrdinalIgnoreCase)) continue;
                        var data = line.Substring(5).Trim();
                        if (data == "[DONE]") break;

                        JObject obj;
                        try { obj = JObject.Parse(data); }
                        catch (JsonException) { continue; }
                        var delta = obj["choices"]?[0]?["delta"];
                        var token = delta?["content"]?.ToString();
                        if (!string.IsNullOrEmpty(token)) yield return ChatStreamResult.Text(token);

                        foreach (var call in delta?["tool_calls"] as JArray ?? new JArray())
                        {
                            var index = call["index"]?.Value<int>() ?? 0;
                            if (!pendingCalls.TryGetValue(index, out var pending))
                                pendingCalls[index] = pending = new PendingToolCall();
                            var function = call["function"];
                            if (!string.IsNullOrEmpty(function?["name"]?.ToString())) pending.Name = function["name"].ToString();
                            pending.Arguments.Append(function?["arguments"]?.ToString() ?? string.Empty);
                        }
                    }
                }

                foreach (var call in pendingCalls.OrderBy(x => x.Key).Select(x => x.Value))
                {
                    Dictionary<string, object> arguments;
                    try { arguments = JsonConvert.DeserializeObject<Dictionary<string, object>>(call.Arguments.ToString()) ?? new Dictionary<string, object>(); }
                    catch { arguments = new Dictionary<string, object>(); }
                    if (!string.IsNullOrWhiteSpace(call.Name)) yield return ChatStreamResult.ToolCall(call.Name, arguments);
                }
            }
        }

        public async Task<string> ChatAsync(string model, List<ChatMessage> messages, LMStudioOptions options = null, CancellationToken ct = default)
        {
            var result = new StringBuilder();
            await foreach (var token in StreamChatAsync(model, messages, options, ct)) result.Append(token);
            return result.ToString();
        }

        // LM Studio's OpenAI-compatible API does not expose backend VRAM telemetry.
        public Task<VramStatus> GetVramStatusAsync(CancellationToken ct = default) =>
            Task.FromResult(new VramStatus { IsHealthy = true });

        private sealed class PendingToolCall
        {
            public string Name { get; set; }
            public StringBuilder Arguments { get; } = new StringBuilder();
        }
    }

    public class ChatStreamResult
    {
        public bool IsTextToken { get; private set; }
        public bool IsToolCall { get; private set; }
        public string TextToken { get; private set; }
        public string ToolName { get; private set; }
        public Dictionary<string, object> ToolArguments { get; private set; }
        public static ChatStreamResult Text(string token) => new ChatStreamResult { IsTextToken = true, TextToken = token };
        public static ChatStreamResult ToolCall(string name, Dictionary<string, object> args) => new ChatStreamResult { IsToolCall = true, ToolName = name, ToolArguments = args };
    }

    public class LMStudioToolDefinition
    {
        [JsonProperty("type")]
        public string Type { get; set; } = "function";
        [JsonProperty("function")]
        public LMStudioFunctionDefinition Function { get; set; }
    }

    public class LMStudioFunctionDefinition
    {
        [JsonProperty("name")]
        public string Name { get; set; }
        [JsonProperty("description")]
        public string Description { get; set; }
        [JsonProperty("parameters")]
        public LMStudioParameterDefinition Parameters { get; set; }
    }

    public class LMStudioParameterDefinition
    {
        [JsonProperty("type")]
        public string Type { get; set; } = "object";
        [JsonProperty("properties")]
        public Dictionary<string, LMStudioPropertyDefinition> Properties { get; set; } = new Dictionary<string, LMStudioPropertyDefinition>();
        [JsonProperty("required")]
        public List<string> Required { get; set; } = new List<string>();
    }

    public class LMStudioPropertyDefinition
    {
        [JsonProperty("type")]
        public string Type { get; set; } = "string";
        [JsonProperty("description")]
        public string Description { get; set; }
    }

    public class LMStudioOptions
    {
        public double Temperature { get; set; } = 0.2;
        public double TopP { get; set; } = 0.9;
        public int TopK { get; set; } = 40;
        public double RepeatPenalty { get; set; } = 1.1;
        public int NumCtx { get; set; } = LocalPilot.Settings.LocalPilotSettings.Instance.ContextWindowSize;
        public int NumPredict { get; set; } = LocalPilot.Settings.LocalPilotSettings.Instance.MaxOutputTokens;
        public List<string> Stop { get; set; } = new List<string>();
        public int RequestTimeoutSeconds { get; set; } = 120;
    }

    public class ChatMessage
    {
        [JsonProperty("role")]
        public string Role { get; set; }
        [JsonProperty("content")]
        public string Content { get; set; }
    }

    public class VramStatus
    {
        public bool IsHealthy { get; set; }
        public int TotalVramUsedMb { get; set; }
        public int ActiveModelsCount { get; set; }
    }
}
