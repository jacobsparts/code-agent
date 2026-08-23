import struct
from dataclasses import dataclass

@dataclass(frozen=True)
class F:
    name: str
    num: int
    type: str = None
    rep: bool = False
    packed: bool = False
    map: tuple = None
    enum: dict = None

MESSAGES = {
    "AvailableModels_req": [
        F("useModelParameters", 5, type="bool"),
        F("doNotUseMarkdown", 7, type="bool"),
    ],
    "AvailableModels_res": [
        F("models", 2, type="M:AvailableModel", rep=True),
        F("composerModelConfig", 4, type="M:ComposerModelConfig"),
        F("cmdKModelConfig", 5, type="M:CmdKModelConfig"),
        F("backgroundComposerModelConfig", 6, type="M:BackgroundComposerModelConfig"),
        F("planExecutionModelConfig", 7, type="M:PlanExecutionModelConfig"),
        F("specModelConfig", 8, type="M:SpecModelConfig"),
        F("deepSearchModelConfig", 9, type="M:DeepSearchModelConfig"),
        F("quickAgentModelConfig", 10, type="M:QuickAgentModelConfig"),
        F("useModelParameters", 11, type="bool"),
        F("disableUnusedModelsAfterNHours", 12, type="int32"),
        F("upgradeUnchangedModelsAfterNHours", 13, type="int32"),
        F("displayConfiguration", 15, type="M:DisplayConfiguration"),
        F("subagentModelConfigs", 16, map=("string", "M:SubagentModelConfig")),
    ],
    "QuickAgentModelConfig": [
        F("defaultModel", 1, type="string"),
    ],
    "DisplayConfiguration": [
        F("routedModelViewConfig", 1, type="M:RoutedModelViewConfig"),
        F("namedModelsViewConfig", 2, type="M:NamedModelsViewConfig"),
    ],
    "RoutedModelViewConfig": [
        F("routedModelViewToNamedViewToggle", 2, type="M:RoutedModelViewToNamedViewToggle"),
        F("hideSearchBar", 4, type="bool"),
    ],
    "RoutedModelViewToNamedViewToggle": [
        F("titleMarkdown", 1, type="string"),
        F("subtitle", 2, type="string"),
        F("setToLastNamedModel", 3, type="bool"),
    ],
    "NamedModelsViewConfig": [
        F("namedViewToRoutedModelViewToggle", 2, type="M:NamedViewToRoutedModelViewToggle"),
    ],
    "NamedViewToRoutedModelViewToggle": [
        F("markdown", 1, type="string"),
    ],
    "SubagentModelConfig": [
        F("defaultModel", 1, type="string"),
        F("fallbackModels", 2, type="string", rep=True),
    ],
    "AvailableModel": [
        F("name", 1, type="string"),
        F("defaultOn", 2, type="bool"),
        F("supportsAgent", 5, type="bool"),
        F("degradationStatus", 6, type="int32", enum={0: "DEGRADATION_STATUS_UNSPECIFIED"}),
        F("tooltipData", 8, type="M:ModelTooltipData"),
        F("supportsThinking", 9, type="bool"),
        F("supportsImages", 10, type="bool"),
        F("supportsMaxMode", 14, type="bool"),
        F("contextTokenLimit", 15, type="int32"),
        F("contextTokenLimitForMaxMode", 16, type="int32"),
        F("clientDisplayName", 17, type="string"),
        F("serverModelName", 18, type="string"),
        F("supportsNonMaxMode", 19, type="bool"),
        F("tooltipDataForMaxMode", 20, type="M:MaxModeTooltipData"),
        F("isRecommendedForBackgroundComposer", 21, type="bool"),
        F("supportsPlanMode", 22, type="bool"),
        F("inputboxShortModelName", 24, type="string"),
        F("supportsSandboxing", 25, type="bool"),
        F("supportsCmdK", 26, type="bool"),
        F("backgroundComposerSortOrder", 28, type="int32"),
        F("parameterDefinitions", 29, type="M:ModelParameterDefinition", rep=True),
        F("variants", 30, type="M:ModelVariant", rep=True),
        F("upgradeModelId", 34, type="string"),
        F("legacySlugs", 36, type="string", rep=True),
        F("idAliases", 37, type="string", rep=True),
        F("namedModelSectionIndex", 38, type="int32"),
        F("tagline", 39, type="string"),
        F("vendorName", 41, type="string"),
        F("vendor", 42, type="M:ModelVendor"),
        F("supportsSmartModeClassifier", 45, type="bool"),
        F("requiresDataRetention", 46, type="bool"),
        F("reasonForZdrConsentBlock", 47, type="string"),
    ],
    "ModelVendor": [
        F("id", 1, type="int32", enum={
            1: "MODEL_VENDOR_ID_ANTHROPIC",
            2: "MODEL_VENDOR_ID_OPENAI",
            3: "MODEL_VENDOR_ID_GOOGLE",
            5: "MODEL_VENDOR_ID_MOONSHOT",
            6: "MODEL_VENDOR_ID_CURSOR",
            8: "MODEL_VENDOR_ID_ZAI",
        }),
        F("displayName", 2, type="string"),
    ],
    "MaxModeTooltipData": [
        F("markdownContent", 7, type="string"),
    ],
    "ModelParameterDefinition": [
        F("id", 1, type="string"),
        F("name", 2, type="string"),
        F("markdownTooltip", 3, type="string"),
        F("parameterType", 4, type="M:ModelParameterType"),
        F("isCycleableByHotkey", 5, type="bool"),
    ],
    "ModelParameterType": [
        F("booleanParameter", 1, type="M:BooleanModelParameter"),
        F("enumParameter", 2, type="M:EnumModelParameter"),
    ],
    "BooleanModelParameter": [
        F("values", 1, type="M:ModelParameterValue", rep=True),
    ],
    "EnumModelParameter": [
        F("values", 1, type="M:ModelParameterValue", rep=True),
    ],
    "ModelParameterValue": [
        F("value", 1, type="string"),
        F("displayName", 2, type="string"),
        F("increasesModelCost", 3, type="bool"),
    ],
    "ModelVariantParameterValue": [
        F("id", 1, type="string"),
        F("value", 2, type="string"),
    ],
    "ModelVariant": [
        F("parameterValues", 1, type="M:ModelVariantParameterValue", rep=True),
        F("displayName", 2, type="string"),
        F("isMaxMode", 3, type="bool"),
        F("isDefaultMaxConfig", 4, type="bool"),
        F("isDefaultNonMaxConfig", 5, type="bool"),
        F("tooltipData", 6, type="M:ModelVariantTooltipData"),
        F("displayNameOutsidePicker", 8, type="string"),
        F("variantStringRepresentation", 9, type="string"),
        F("confirmationDialogue", 10, type="M:ModelVariantConfirmationDialogue"),
        F("legacySlug", 11, type="string"),
    ],
    "ModelVariantConfirmationDialogue": [
        F("title", 1, type="string"),
        F("body", 2, type="string"),
        F("key", 3, type="string"),
    ],
    "ModelVariantTooltipData": [
        F("markdownContent", 7, type="string"),
    ],
    "ModelTooltipData": [
        F("markdownContent", 7, type="string"),
    ],
    "ComposerModelConfig": [
        F("defaultModel", 1, type="string"),
        F("fallbackModels", 2, type="string", rep=True),
        F("bestOfNDefaultModels", 3, type="string", rep=True),
    ],
    "CmdKModelConfig": [
        F("defaultModel", 1, type="string"),
    ],
    "BackgroundComposerModelConfig": [
        F("defaultModel", 1, type="string"),
        F("fallbackModels", 2, type="string", rep=True),
        F("bestOfNDefaultModels", 3, type="string", rep=True),
    ],
    "PlanExecutionModelConfig": [
        F("defaultModel", 1, type="string"),
        F("fallbackModels", 2, type="string", rep=True),
    ],
    "SpecModelConfig": [
        F("defaultModel", 1, type="string"),
    ],
    "DeepSearchModelConfig": [
        F("defaultModel", 1, type="string"),
    ],
    "GetCliDownloadUrl_req": [
        F("channel", 2, type="string"),
    ],
    "GetCliDownloadUrl_res": [
        F("url", 1, type="string"),
        F("version", 2, type="string"),
    ],
    "GetDefaultModelForCli_res": [
        F("model", 1, type="M:CliModel"),
    ],
    "CliModel": [
        F("modelId", 1, type="string"),
        F("displayModelId", 3, type="string"),
        F("displayName", 4, type="string"),
        F("displayNameShort", 5, type="string"),
        F("aliases", 6, type="string", rep=True),
        F("maxMode", 7, type="bool"),
    ],
    "GetEffectiveUserPlugins_res": [
        F("marketplaces", 2, type="M:EffectiveUserPluginMarketplace", rep=True),
    ],
    "EffectiveUserPluginMarketplace": [
        F("id", 1, type="int64"),
        F("name", 2, type="string"),
        F("displayName", 3, type="string"),
        F("description", 4, type="string"),
        F("createdAt", 10, type="int64"),
        F("updatedAt", 11, type="int64"),
        F("allowUserPublish", 19, type="bool"),
    ],
    "GetGlobalCommands_req": [
        F("surface", 1, type="int32", enum={3: 'GLOBAL_COMMAND_SURFACE_CLI'}),
    ],
    "GetGlobalCommands_res": [
        F("commands", 1, type="M:GlobalCommand", rep=True),
    ],
    "GlobalCommand": [
        F("name", 1, type="string"),
        F("content", 2, type="string"),
        F("description", 3, type="string"),
        F("availability", 4, type="int32", enum={1: "GLOBAL_COMMAND_AVAILABILITY_ALL_AGENTS", 3: "GLOBAL_COMMAND_AVAILABILITY_LOCAL_ONLY"}),
        F("argumentHint", 5, type="string"),
        F("disabledSurfaces", 6, type="int32", packed=True, enum={2: "GLOBAL_COMMAND_SURFACE_GLASS"}),
        F("requiresAllWorkspaceFoldersAreGitRepos", 7, type="bool"),
    ],
    "GetManagedSkills_res": [
        F("skills", 1, type="M:ManagedSkill", rep=True),
    ],
    "ManagedSkill": [
        F("id", 1, type="string"),
        F("description", 2, type="string"),
        F("content", 3, type="string"),
        F("disableModelInvocation", 4, type="bool"),
        F("environments", 5, type="string", rep=True),
        F("disabledEnvironments", 6, type="string", rep=True),
        F("enabled", 7, type="bool"),
        F("resources", 9, map=("string", "string")),
    ],
    "GetMe_res": [
        F("authId", 1, type="string"),
        F("userId", 2, type="int32"),
        F("email", 3, type="string"),
        F("firstName", 4, type="string"),
        F("lastName", 5, type="string"),
        F("workosId", 6, type="string"),
        F("createdAt", 8, type="string"),
        F("isEnterpriseUser", 9, type="bool"),
        F("country", 12, type="string"),
        F("cursorReviewOnboardingUseCursorGithubApp", 14, type="bool"),
        F("f19", 19, type="string"),
    ],
    "GetServerConfig_res": [
        F("bugConfigResponse", 1, type="M:BugConfigResponse"),
        F("indexingConfig", 3, type="M:IndexingConfig"),
        F("clientTracingConfig", 4, type="M:ClientTracingConfig"),
        F("chatConfig", 5, type="M:ChatConfig"),
        F("configVersion", 6, type="string"),
        F("profilingConfig", 8, type="M:Empty"),
        F("metricsConfig", 9, type="M:MetricsConfig"),
        F("backgroundComposerConfig", 10, type="M:BackgroundComposerConfig"),
        F("memoryMonitorConfig", 13, type="M:MemoryMonitorConfig"),
        F("folderSizeLimit", 14, type="M:FolderSizeLimit"),
        F("gitIndexingConfig", 15, type="M:GitIndexingConfig"),
        F("currentInAppAd", 17, type="M:CurrentInAppAd"),
        F("traceConfig", 18, type="M:TraceConfig"),
        F("runTerminalServerConfig", 19, type="M:RunTerminalServerConfig"),
        F("onlineMetricsConfig", 20, type="M:OnlineMetricsConfig"),
        F("interactionConfig", 21, type="M:InteractionConfig"),
        F("agentTelemetryConfig", 22, type="M:AgentTelemetryConfig"),
        F("clientVersionStatus", 23, type="M:ClientVersionStatus"),
        F("agentLayoutPolicy", 25, type="M:AgentLayoutPolicy"),
        F("useNlbForNal", 26, type="bool"),
        F("agentUrlConfig", 27, type="M:AgentUrlConfig"),
        F("cliSandboxDefaultEnabled", 28, type="bool"),
        F("onboardingConfig", 33, type="M:OnboardingConfig"),
        F("bugbotConfig", 34, type="M:Empty"),
        F("codebaseTelemetryConfig", 35, type="M:CodebaseTelemetryConfig"),
        F("googleWorkspaceMcpOauthClient", 37, type="M:GoogleWorkspaceMcpOauthClient"),
    ],
    "GoogleWorkspaceMcpOauthClient": [
        F("clientId", 1, type="string"),
        F("clientSecret", 2, type="string"),
    ],
    "AgentUrlConfig": [
        F("agentUrl", 1, type="string"),
        F("agentnUrl", 2, type="string"),
    ],
    "AgentLayoutPolicy": [
        F("allowedActionIds", 1, type="string", rep=True),
        F("deniedActionIds", 2, type="string", rep=True),
    ],
    "BugConfigResponse": [
        F("bugBotV1", 2, type="M:BugBotV1"),
    ],
    "BackgroundComposerConfig": [
        F("showBackgroundAgentInBetaSettings", 2, type="bool"),
        F("windowInWindowPreloadCount", 3, type="int32"),
        F("windowInWindowPingIntervalMs", 4, type="double"),
        F("showBackgroundAgentDisclaimer", 5, type="bool"),
        F("showBackgroundAgentSlackAd", 6, type="bool"),
        F("showBackgroundAgentHistoryAction", 7, type="bool"),
        F("maxWindowInWindows", 8, type="int32"),
    ],
    "MemoryMonitorConfig": [
        F("baseThresholdMb", 3, type="int32"),
        F("criticalOffsetMb", 4, type="int32"),
        F("processMemoryIntervalSec", 5, type="int32"),
    ],
    "FolderSizeLimit": [
        F("maxTotalBytes", 1, type="int32"),
        F("maxNumFiles", 2, type="int32"),
    ],
    "GitIndexingConfig": [
        F("enabled", 1, type="bool"),
    ],
    "CurrentInAppAd": [
        F("id", 1, type="string"),
        F("header", 2, type="M:InAppAdHeader"),
        F("content", 3, type="M:InAppAdContent"),
        F("buttons", 4, type="M:InAppAdButton", rep=True),
        F("displayMode", 6, type="int32", enum={1: "DISPLAY_MODE_TOAST"}),
        F("targetSurfaces", 9, type="int32", packed=True, enum={1: "CLIENT_SURFACE_EDITOR", 2: "CLIENT_SURFACE_GLASS"}),
    ],
    "InAppAdHeader": [
        F("bannerUrl", 1, type="string"),
        F("bannerUrlLight", 2, type="string"),
        F("bannerUrlDark", 3, type="string"),
    ],
    "InAppAdContent": [
        F("title", 2, type="string"),
        F("sections", 3, type="M:InAppAdSection", rep=True),
    ],
    "InAppAdSection": [
        F("iconSvg", 1, type="string"),
        F("title", 2, type="string"),
        F("description", 3, type="string"),
    ],
    "InAppAdButton": [
        F("text", 1, type="string"),
        F("buttonType", 2, type="int32", enum={2: 'BUTTON_TYPE_PRIMARY'}),
        F("externalUrl", 3, type="string"),
    ],
    "TraceConfig": [
        F("enabled", 1, type="bool"),
        F("bufferSize", 2, type="int32"),
        F("flushIntervalMs", 3, type="int32"),
        F("sampleRate", 4, type="double"),
        F("internalSampleRate", 5, type="double"),
        F("internalExtHostSampleRate", 6, type="double"),
    ],
    "RunTerminalServerConfig": [
        F("compositeShellCommands", 1, type="string", rep=True),
    ],
    "BugBotV1": [
        F("backgroundCallFrequencyMs", 3, type="int32"),
    ],
    "OnlineMetricsConfig": [
        F("enabled", 1, type="bool"),
        F("maxRequestsTracked", 2, type="int32"),
        F("maxRequestsTrackedMb", 3, type="int32"),
        F("maxRequestRetentionSeconds", 4, type="double"),
        F("numCommitsTracked", 7, type="int32"),
        F("timeIntervalsTrackedMinutes", 8, type="int32", packed=True),
        F("tooBigFileSizeBytes", 9, type="int32"),
    ],
    "InteractionConfig": [
        F("metricsEnabled", 2, type="bool"),
        F("profilingIntervalSec", 3, type="int32"),
        F("metricsIntervalSec", 4, type="int32"),
        F("profilingMaxBufferSize", 5, type="int32"),
        F("profilingInteractionDurationThresholdMs", 6, type="int32"),
        F("profilingSampleIntervalMs", 7, type="int32"),
        F("metricsMinInteractionsForLoaf", 8, type="int32"),
        F("metricsMinForegroundTimeMs", 9, type="int32"),
        F("metricsCombinedInpDropHighestCount", 10, type="int32"),
        F("metricsClickInpDropHighestCount", 11, type="int32"),
        F("metricsKeypressInpDropHighestCount", 12, type="int32"),
        F("metricsStartupThresholdSec", 13, type="int32"),
        F("analyticsSummaryEveryNWindows", 14, type="int32"),
        F("extHostLagSummaryIntervalSec", 15, type="int32"),
    ],
    "AgentTelemetryConfig": [
        F("enabled", 1, type="bool"),
    ],
    "ClientVersionStatus": [
        F("updateLevel", 1, type="int32", enum={3: 'CLIENT_UPDATE_LEVEL_REQUIRED'}),
        F("currentClientVersion", 2, type="string"),
        F("minSupportedClientVersion", 3, type="string"),
        F("minAllowedClientVersion", 4, type="string"),
        F("message", 5, type="string"),
    ],
    "IndexingConfig": [
        F("maxConcurrentUploads", 1, type="int32"),
        F("absoluteMaxNumberFiles", 2, type="int32"),
        F("maxFileRetries", 3, type="int32"),
        F("syncConcurrency", 4, type="int32"),
        F("autoIndexingMaxNumFiles", 5, type="int32"),
        F("indexingPeriodSeconds", 6, type="int32"),
        F("incremental", 8, type="bool"),
        F("defaultUserPathEncryptionKey", 9, type="string"),
        F("multiRootIndexingEnabled", 11, type="bool"),
        F("copyStatusCheckPeriodSeconds", 12, type="double"),
        F("copyTimeoutSeconds", 13, type="int32"),
        F("maxBatchBytes", 14, type="int32"),
        F("maxBatchNumRequests", 15, type="int32"),
        F("maxSyncMerkleBatchSize", 16, type="int32"),
    ],
    "OnboardingConfig": [
        F("marketplacePluginNames", 1, type="string", rep=True),
    ],
    "CodebaseTelemetryConfig": [
        F("enabled", 1, type="bool"),
    ],
    "ClientTracingConfig": [
        F("globalSampleRate", 1, type="double"),
        F("tracesSampleRate", 2, type="double"),
        F("loggerSampleRate", 3, type="double"),
        F("minidumpSampleRate", 4, type="double"),
        F("errorRateLimit", 5, type="double"),
        F("performanceUnitRateLimit", 6, type="double"),
        F("profilesSampleRate", 7, type="double"),
        F("jsonStringifySampleRate", 8, type="double"),
    ],
    "ChatConfig": [
        F("fullContextTokenLimit", 2, type="int32"),
        F("maxRuleLength", 4, type="int32"),
        F("maxMcpTools", 5, type="int32"),
        F("warnMcpTools", 6, type="int32"),
        F("summarizationMessage", 7, type="string"),
        F("numSummarizationsBeforeWarningShown", 10, type="int32"),
        F("cursorRulesReadFileFixEnabled", 11, type="bool"),
        F("dontSendCtrlCBeforeCommand", 12, type="bool"),
        F("clientStatsigPollIntervalMs", 13, type="int32"),
        F("listDirV2PredefinedIgnoreGlobs", 15, type="string", rep=True),
    ],
    "MetricsConfig": [
        F("enabledInPrivacyMode", 2, type="bool"),
        F("enabledInNonPrivacyMode", 3, type="bool"),
    ],
    "GetUsableModels_res": [
        F("models", 1, type="M:UsableModel", rep=True),
    ],
    "UsableModel": [
        F("modelId", 1, type="string"),
        F("displayModelId", 3, type="string"),
        F("displayName", 4, type="string"),
        F("displayNameShort", 5, type="string"),
        F("aliases", 6, type="string", rep=True),
        F("maxMode", 7, type="bool"),
    ],
    "GetUserPrivacyMode_res": [
        F("privacyMode", 1, type="int32", enum={2: 'PRIVACY_MODE_NO_TRAINING'}),
    ],
    "ListMarketplaces_res": [
        F("marketplaces", 1, type="M:Marketplace", rep=True),
    ],
    "Marketplace": [
        F("id", 1, type="int64"),
        F("name", 2, type="string"),
        F("displayName", 3, type="string"),
        F("description", 4, type="string"),
        F("createdAt", 10, type="int64"),
        F("updatedAt", 11, type="int64"),
        F("allowUserPublish", 19, type="bool"),
    ],
    "NameAgent_req": [
        F("userMessage", 1, type="string"),
    ],
    "NameAgent_res": [
        F("name", 1, type="string"),
    ],
    "ReportDistribution_req": [
        F("metricsList", 2, type="M:DistributionMetric", rep=True),
    ],
    "DistributionMetric": [
        F("name", 1, type="string"),
        F("value", 2, type="double"),
        F("tags", 3, map=("string", "string")),
    ],
    "ReportIncrement_req": [
        F("metricsList", 2, type="M:IncrementMetric", rep=True),
    ],
    "IncrementMetric": [
        F("name", 1, type="string"),
        F("value", 2, type="double"),
        F("tags", 3, map=("string", "string")),
    ],
    "Run_res": [
        F("interactionUpdate", 1, type="M:InteractionUpdate"),
        F("execServerMessage", 2, type="M:ExecServerMessage"),
        F("conversationCheckpointUpdate", 3, type="M:ConversationCheckpointUpdate"),
        F("kvServerMessage", 4, type="M:KvServerMessage"),
        F("ttftBreakdown", 8, type="M:TtftBreakdown"),
    ],
    "InteractionUpdate": [
        F("textDelta", 1, type="M:TextDelta"),
        F("toolCallStarted", 2, type="M:ToolCallStarted"),
        F("toolCallCompleted", 3, type="M:ToolCallCompleted"),
        F("thinkingDelta", 4, type="M:ThinkingDelta"),
        F("thinkingCompleted", 5, type="M:ThinkingCompleted"),
        F("partialToolCall", 7, type="M:PartialToolCall"),
        F("tokenDelta", 8, type="M:TokenDelta"),
        F("heartbeat", 13, type="M:Empty"),
        F("turnEnded", 14, type="M:TurnEnded"),
        F("stepCompleted", 17, type="M:StepCompleted"),
        F("toolCallDelta", 15, type="M:ToolCallDelta"),
        F("feedbackRequest", 21, type="M:FeedbackRequest"),
        F("f25", 25, type="int64"),
    ],
    "TextDelta": [
        F("text", 1, type="string"),
    ],
    "TurnEnded": [
        F("inputTokens", 1, type="int64"),
        F("outputTokens", 2, type="int64"),
        F("cacheReadTokens", 3, type="int64"),
        F("cacheWriteTokens", 4, type="int64"),
        F("reasoningTokens", 5, type="int64"),
    ],
    "ToolCallDelta": [
        F("callId", 1, type="string"),
        F("toolCallDelta", 2, type="M:ToolCallDeltaValue"),
        F("modelCallId", 3, type="string"),
    ],
    "ToolCallDeltaValue": [
        F("shellToolCallDelta", 1, type="M:ShellToolCallDelta"),
    ],
    "ShellToolCallDelta": [
        F("stdout", 1, type="M:ShellToolCallStdout"),
    ],
    "ShellToolCallStdout": [
        F("content", 1, type="string"),
    ],
    "StepCompleted": [
        F("stepId", 1, type="int64"),
        F("stepDurationMs", 2, type="int64"),
    ],
    "ToolCallStarted": [
        F("callId", 1, type="string"),
        F("toolCall", 2, type="M:StartedToolCall"),
        F("modelCallId", 3, type="string"),
    ],
    "StartedToolCall": [
        F("shellToolCall", 1, type="M:ShellToolCall"),
        F("globToolCall", 4, type="M:StartedGlobToolCall"),
        F("grepToolCall", 5, type="M:StartedGrepToolCall"),
        F("readToolCall", 8, type="M:StartedReadToolCall"),
        F("getMcpToolsToolCall", 44, type="M:McpToolsToolCall"),
        F("toolCallId", 57, type="string"),
        F("startedAtMs", 59, type="int64"),
    ],
    "StartedGlobToolCall": [
        F("args", 1, type="M:GlobToolCallArgs"),
    ],
    "GlobToolCallArgs": [
        F("targetDirectory", 1, type="string"),
        F("globPattern", 2, type="string"),
    ],
    "StartedGrepToolCall": [
        F("args", 1, type="M:GrepToolCallArgs"),
    ],
    "GrepToolCallArgs": [
        F("pattern", 1, type="string"),
        F("path", 2, type="string"),
        F("glob", 3, type="string"),
        F("contextAfter", 6, type="int32"),
        F("caseInsensitive", 8, type="bool"),
        F("headLimit", 10, type="int32"),
        F("multiline", 11, type="bool"),
        F("toolCallId", 14, type="string"),
        F("offset", 16, type="int32"),
    ],
    "StartedReadToolCall": [
        F("args", 1, type="M:ReadToolCallArgs"),
    ],
    "ReadToolCallArgs": [
        F("path", 1, type="string"),
        F("offset", 2, type="int32"),
        F("limit", 3, type="int32"),
    ],
    "ToolCallCompleted": [
        F("callId", 1, type="string"),
        F("toolCall", 2, type="M:CompletedToolCall"),
        F("modelCallId", 3, type="string"),
    ],
    "CompletedToolCall": [
        F("shellToolCall", 1, type="M:ShellToolCall"),
        F("globToolCall", 4, type="M:CompletedGlobToolCall"),
        F("grepToolCall", 5, type="M:CompletedGrepToolCall"),
        F("readToolCall", 8, type="M:CompletedReadToolCall"),
        F("getMcpToolsToolCall", 44, type="M:McpToolsToolCall"),
        F("toolCallId", 57, type="string"),
        F("startedAtMs", 59, type="int64"),
        F("completedAtMs", 60, type="int64"),
    ],
    "CompletedGlobToolCall": [
        F("args", 1, type="M:CompletedGlobToolCallArgs"),
        F("result", 2, type="M:GlobToolCallResult"),
    ],
    "CompletedGlobToolCallArgs": [
        F("targetDirectory", 1, type="string"),
        F("globPattern", 2, type="string"),
    ],
    "GlobToolCallResult": [
        F("success", 1, type="M:GlobToolCallSuccess"),
    ],
    "GlobToolCallSuccess": [
        F("path", 2, type="string"),
        F("files", 3, type="string", rep=True),
        F("totalFiles", 4, type="int32"),
    ],
    "CompletedGrepToolCall": [
        F("args", 1, type="M:CompletedGrepToolCallArgs"),
        F("result", 2, type="M:GrepToolCallResult"),
    ],
    "CompletedGrepToolCallArgs": [
        F("pattern", 1, type="string"),
        F("path", 2, type="string"),
        F("glob", 3, type="string"),
        F("contextAfter", 6, type="int32"),
        F("caseInsensitive", 8, type="bool"),
        F("headLimit", 10, type="int32"),
        F("multiline", 11, type="bool"),
        F("toolCallId", 14, type="string"),
        F("offset", 16, type="int32"),
    ],
    "GrepToolCallResult": [
        F("success", 1, type="M:GrepToolCallSuccess"),
    ],
    "GrepToolCallSuccess": [
        F("pattern", 1, type="string"),
        F("path", 2, type="string"),
        F("outputMode", 3, type="string"),
        F("workspaceResults", 4, map=("string", "M:GrepWorkspaceResult")),
    ],
    "GrepWorkspaceResults": [
        F("workspaceResults", 1, map=("string", "M:GrepWorkspaceResult")),
    ],
    "GrepWorkspaceResult": [
        F("files", 2, type="M:GrepFiles"),
        F("content", 3, type="M:GrepContent"),
    ],
    "GrepFiles": [
        F("files", 1, type="string", rep=True),
        F("totalFiles", 2, type="int32"),
    ],
    "GrepContent": [
        F("matches", 1, type="M:GrepFileMatches", rep=True),
        F("totalLines", 2, type="int32"),
        F("totalMatchedLines", 3, type="int32"),
        F("clientTruncated", 4, type="bool"),
        F("headLimitApplied", 6, type="int32"),
    ],
    "GrepFileMatches": [
        F("file", 1, type="string"),
        F("matches", 2, type="M:GrepLineMatch", rep=True),
    ],
    "GrepLineMatch": [
        F("lineNumber", 1, type="int32"),
        F("content", 2, type="string"),
        F("isContextLine", 4, type="bool"),
    ],
    "CompletedReadToolCall": [
        F("args", 1, type="M:CompletedReadToolCallArgs"),
        F("result", 2, type="M:ReadToolCallResult"),
    ],
    "CompletedReadToolCallArgs": [
        F("path", 1, type="string"),
        F("offset", 2, type="int32"),
        F("limit", 3, type="int32"),
    ],
    "ReadToolCallResult": [
        F("success", 1, type="M:ReadToolCallSuccess"),
        F("error", 2, type="M:ReadToolCallError"),
    ],
    "ReadToolCallError": [
        F("errorMessage", 1, type="string"),
    ],
    "ReadToolCallSuccess": [
        F("content", 1, type="string"),
        F("totalLines", 4, type="int32"),
        F("fileSize", 5, type="int32"),
        F("path", 7, type="string"),
        F("readRange", 8, type="M:ReadRange"),
    ],
    "ReadRange": [
        F("startLine", 1, type="int32"),
        F("endLine", 2, type="int32"),
    ],
    "ThinkingDelta": [
        F("text", 1, type="string"),
        F("thinkingStyle", 2, type="int32", enum={1: "THINKING_STYLE_DEFAULT"}),
    ],
    "ThinkingCompleted": [
        F("thinkingDurationMs", 1, type="int32"),
    ],
    "PartialToolCall": [
        F("callId", 1, type="string"),
        F("toolCall", 2, type="M:PartialToolCallValue"),
        F("modelCallId", 4, type="string"),
    ],
    "PartialToolCallValue": [
        F("globToolCall", 4, type="M:Empty"),
        F("grepToolCall", 5, type="M:Empty"),
        F("readToolCall", 8, type="M:Empty"),
        F("getMcpToolsToolCall", 44, type="M:McpToolsToolCall"),
        F("toolCallId", 57, type="string"),
        F("startedAtMs", 59, type="int64"),
    ],
    "TokenDelta": [
        F("tokens", 1, type="int32"),
    ],
    "ShellToolCall": [
        F("args", 1, type="M:ShellArgs"),
        F("result", 2, type="M:ShellResult"),
        F("description", 3, type="string"),
    ],
    "ShellArgs": [
        F("command", 1, type="string"),
        F("timeout", 3, type="int32"),
        F("toolCallId", 4, type="string"),
        F("simpleCommands", 5, type="string", rep=True),
        F("parsingResult", 8, type="M:ShellParsingResult"),
        F("fileOutputThresholdBytes", 10, type="int64"),
        F("timeoutBehavior", 13, type="int32", enum={
            2: "TIMEOUT_BEHAVIOR_BACKGROUND",
        }),
        F("hardTimeout", 14, type="int32"),
        F("description", 15, type="string"),
        F("closeStdin", 17, type="bool"),
        F("conversationId", 21, type="string"),
    ],
    "ShellParsingResult": [
        F("executableCommands", 2, type="M:ExecutableCommand", rep=True),
    ],
    "ExecutableCommand": [
        F("name", 1, type="string"),
        F("fullText", 3, type="string"),
    ],
    "ShellResult": [
        F("success", 1, type="M:ShellSuccess"),
        F("isBackground", 102, type="bool"),
    ],
    "ShellSuccess": [
        F("command", 1, type="string"),
        F("stdout", 5, type="string"),
        F("executionTime", 7, type="int32"),
        F("interleavedOutput", 10, type="string"),
        F("localExecutionTimeMs", 13, type="int32"),
    ],

    "ExecServerMessage": [
        F("id", 1, type="int32"),
        F("grepArgs", 5, type="M:GrepArgs"),
        F("readArgs", 7, type="M:ReadArgs"),
        F("shellStreamArgs", 14, type="M:ShellArgs"),
        F("execId", 15, type="string"),
        F("spanContext", 19, type="M:ExecSpanContext"),
        F("mcpStateExecArgs", 36, type="M:McpStateExecArgs"),
        F("acceptHookAdditionalContexts", 55, type="bool"),
    ],
    "GrepArgs": [
        F("pattern", 1, type="string"),
        F("path", 2, type="string"),
        F("glob", 3, type="string"),
        F("outputMode", 4, type="string"),
        F("contextBefore", 5, type="int32"),
        F("contextAfter", 6, type="int32"),
        F("context", 7, type="int32"),
        F("caseInsensitive", 8, type="bool"),
        F("type", 9, type="string"),
        F("headLimit", 10, type="int32"),
        F("multiline", 11, type="bool"),
        F("sort", 12, type="string"),
        F("sortAscending", 13, type="bool"),
        F("toolCallId", 14, type="string"),
        F("sandboxPolicy", 15, type="bytes"),
        F("offset", 16, type="int32"),
    ],
    "ExecSpanContext": [
        F("traceId", 1, type="string"),
        F("spanId", 2, type="string"),
        F("traceFlags", 3, type="int32"),
    ],
    "ReadArgs": [
        F("path", 1, type="string"),
        F("toolCallId", 2, type="string"),
        F("offset", 4, type="int32"),
        F("limit", 5, type="int32"),
        F("encodingHint", 6, type="string"),
    ],
    "ConversationCheckpointUpdate": [
        F("rootPromptMessagesJson", 1, type="bytes", rep=True),
        F("pendingToolCalls", 4, type="string", rep=True),
        F("tokenDetails", 5, type="M:CheckpointTokenDetails"),
        F("turns", 8, type="bytes", rep=True),
        F("previousWorkspaceUris", 9, type="string", rep=True),
        F("mode", 10, type="int32", enum={
            1: "AGENT_MODE_AGENT", 2: "AGENT_MODE_ASK",
        }),
        F("readPaths", 18, type="string", rep=True),
        F("agentType", 22, type="string"),
        F("conversationStartedTimestampMs", 26, type="int64"),
        F("conversationStartedTimeZone", 27, type="string"),
    ],
    "CheckpointTokenDetails": [
        F("usedTokens", 1, type="int32"),
        F("maxTokens", 2, type="int32"),
        F("breakdown", 3, type="M:TokenBreakdown"),
    ],
    "TokenBreakdown": [
        F("totalUsedTokens", 1, type="int32"),
        F("maxTokens", 2, type="int32"),
        F("categories", 3, type="M:TokenCategory", rep=True),
    ],
    "TokenCategory": [
        F("id", 1, type="string"),
        F("label", 2, type="string"),
        F("estimatedTokens", 3, type="int32"),
        F("characterCount", 4, type="int32"),
    ],
    "KvServerMessage": [
        F("id", 1, type="int32"),
        F("setBlobArgs", 3, type="M:SetBlobArgs"),
        F("spanContext", 4, type="M:KvSpanContext"),
    ],
    "SetBlobArgs": [
        F("blobId", 1, type="bytes"),
        F("blobData", 2, type="bytes"),
    ],
    "KvSpanContext": [
        F("traceId", 1, type="string"),
        F("spanId", 2, type="string"),
        F("traceFlags", 3, type="int32"),
    ],
    "TtftBreakdown": [
        F("serverFirstTokenMs", 1, type="double"),
        F("preStreamSetupMs", 2, type="double"),
        F("waitForFirstEventMs", 3, type="double"),
        F("providerTtftMs", 4, type="double"),
    ],
    "AgentClientMessage": [
        F("runRequest", 1, type="M:RunRequest"),
        F("execClientMessage", 2, type="M:ExecClientMessage"),
        F("kvClientMessage", 3, type="M:KvClientMessage"),
        F("execClientControlMessage", 5, type="M:ExecClientControlMessage"),
        F("clientHeartbeat", 7, type="M:Empty"),
    ],
    "RunRequest": [
        F("conversationState", 1, type="M:ConversationCheckpointUpdate"),
        F("action", 2, type="M:RunRequestAction"),
        F("mcpTools", 4, type="M:Empty"),
        F("conversationId", 5, type="string"),
        F("requestedModel", 9, type="M:RequestedModel"),
        F("suggestNextPrompt", 10, type="bool"),
        F("excludeWorkspaceContext", 12, type="bool"),
        F("selectedSubagentModels", 14, type="M:RequestedModel", rep=True),
        F("conversationGroupId", 16, type="string"),
        F("runId", 25, type="string"),
    ],
    "RunRequestAction": [
        F("userMessageAction", 1, type="M:UserMessageAction"),
    ],
    "UserMessageAction": [
        F("userMessage", 1, type="M:UserMessage"),
        F("requestContext", 2, type="M:RequestContext"),
    ],
    "UserMessage": [
        F("text", 1, type="string"),
        F("messageId", 2, type="string"),
        F("selectedContext", 3, type="M:Empty"),
        F("mode", 4, type="int32", enum={
            1: "AGENT_MODE_AGENT", 2: "AGENT_MODE_ASK",
        }),
    ],
    "RequestContext": [
        F("env", 4, type="M:Environment"),
        F("repositoryInfo", 6, type="M:RepositoryInfo", rep=True),
        F("sharedNotesListing", 9, type="string"),
        F("commitAttributionMessage", 26, type="string"),
        F("prAttributionMessage", 27, type="string"),
        F("hooksConfig", 28, type="M:Empty"),
        F("agentSkills", 29, type="M:AgentSkill", rep=True),
        F("supportsMcpAuth", 32, type="bool"),
        F("gitRepoInfoComplete", 33, type="bool"),
        F("mcpMetaToolOptions", 34, type="M:McpMetaToolOptions"),
    ],
    "McpMetaToolOptions": [
        F("enabled", 1, type="bool"),
    ],
    "Environment": [
        F("osVersion", 1, type="string"),
        F("workspacePaths", 2, type="string", rep=True),
        F("shell", 3, type="string"),
        F("terminalsFolder", 7, type="string"),
        F("agentSharedNotesFolder", 8, type="string"),
        F("timeZone", 10, type="string"),
        F("projectFolder", 11, type="string"),
        F("agentTranscriptsFolder", 12, type="string"),
        F("sandboxSupported", 14, type="bool"),
        F("sandboxNetworkHasDefaults", 16, type="bool"),
        F("computerUseSupported", 19, type="bool"),
        F("isWorkingDirHomeDir", 20, type="bool"),
        F("processWorkingDirectory", 21, type="string"),
        F("smartModeClassifierAutoModeEnabled", 22, type="bool"),
    ],
    "RepositoryInfo": [
        F("relativeWorkspacePath", 1, type="string"),
        F("repoName", 4, type="string"),
        F("repoOwner", 5, type="string"),
        F("orthogonalTransformSeed", 8, type="double"),
        F("pathEncryptionKey", 10, type="string"),
    ],
    "AgentSkill": [
        F("fullPath", 1, type="string"),
        F("description", 3, type="string"),
        F("environments", 5, type="string", rep=True),
        F("disabledEnvironments", 6, type="string", rep=True),
        F("disableModelInvocation", 8, type="bool"),
    ],
    "RequestedModel": [
        F("modelId", 1, type="string"),
        F("parameters", 3, type="M:RequestedModelParameter", rep=True),
    ],
    "RequestedModelParameter": [
        F("id", 1, type="string"),
        F("value", 2, type="string"),
    ],
    "KvClientMessage": [
        F("id", 1, type="int32"),
        F("setBlobResult", 3, type="M:Empty"),
    ],
    "ExecClientControlMessage": [
        F("streamClose", 1, type="M:StreamClose"),
        F("heartbeat", 3, type="M:Empty"),
    ],
    "StreamClose": [
        F("id", 1, type="int32"),
    ],
    "ExecClientMessage": [
        F("id", 1, type="int32"),
        F("grepResult", 5, type="M:GrepResult"),
        F("readResult", 7, type="M:ReadResult"),
        F("shellStream", 14, type="M:ShellStream"),
        F("localExecutionTimeMs", 39, type="int32"),
    ],
    "ReadResult": [
        F("success", 1, type="M:ReadSuccess"),
        F("error", 2, type="bytes"),
    ],
    "ReadSuccess": [
        F("path", 1, type="string"),
        F("content", 2, type="string"),
        F("totalLines", 3, type="int32"),
        F("fileSize", 4, type="int64"),
        F("rangeApplied", 8, type="bool"),
    ],
    "GrepResult": [
        F("success", 1, type="M:GrepSuccess"),
        F("error", 2, type="bytes"),
    ],
    "GrepSuccess": [
        F("pattern", 1, type="string"),
        F("path", 2, type="string"),
        F("outputMode", 3, type="string"),
        F("workspaceResults", 4, map=("string", "M:GrepWorkspaceResult")),
    ],
    "ShellStream": [
        F("stdout", 1, type="M:ShellData"),
        F("exit", 3, type="M:ShellExit"),
        F("start", 4, type="M:ShellStart"),
    ],
    "ShellData": [
        F("data", 1, type="string"),
    ],
    "ShellExit": [
        F("cwd", 2, type="string"),
        F("localExecutionTimeMs", 6, type="int32"),
    ],
    "ShellStart": [
        F("sandboxPolicy", 1, type="M:SandboxPolicy"),
    ],
    "SandboxPolicy": [
        F("type", 1, type="int32", enum={1: "TYPE_INSECURE_NONE"}),
    ],
    "Empty": [],

    "SubmitLogs_req": [
        F("logs", 1, type="M:LogEntry", rep=True),
    ],
    "LogEntry": [
        F("level", 1, type="int32", enum={1: "CLIENT_LOG_LEVEL_INFO", 2: "CLIENT_LOG_LEVEL_WARN", 3: "CLIENT_LOG_LEVEL_ERROR"}),
        F("message", 2, type="string"),
        F("metadata", 3, map=("string", "string")),
        F("timestamp", 4, type="int64"),
        F("key", 7, type="string"),
    ],
    "SubmitLogs_res": [
        F("success", 1, type="bool"),
        F("logsProcessed", 3, type="int32"),
        F("logsDropped", 4, type="int32"),
    ],
    "TrackEvents_req": [
        F("events", 1, type="M:TrackedEvent", rep=True),
    ],
    "TrackedEvent": [
        F("eventName", 1, type="string"),
        F("eventData", 2, map=("string", "M:EventDataValue")),
        F("timestamp", 3, type="int64"),
    ],
    "EventDataValue": [
        F("stringValue", 1, type="string"),
        F("boolValue", 3, type="bool"),
        F("doubleValue", 4, type="double"),
    ],
    "FeedbackCategory": [
        F("id", 1, type="string"),
        F("label", 2, type="string"),
    ],
    "FeedbackCategoryGroup": [
        F("id", 1, type="string"),
        F("prompt", 2, type="string"),
        F("categories", 3, type="M:FeedbackCategory", rep=True),
    ],
    "FeedbackRequest": [
        F("requestId", 1, type="string"),
        F("canonicalModelName", 2, type="string"),
        F("categories", 3, type="M:FeedbackCategory", rep=True),
        F("categoryGroups", 4, type="M:FeedbackCategoryGroup", rep=True),
        F("title", 6, type="string"),
        F("negativeTitle", 7, type="string"),
        F("commentPlaceholder", 8, type="string"),
    ],
    "McpStateExecArgs": [
        F("args", 1, map=("string", "string")),
    ],
    "McpToolsToolCall": [
        F("args", 1, type="M:McpToolsToolCallArgs"),
        F("result", 2, type="M:McpToolsToolCallResult"),
    ],
    "McpToolsToolCallArgs": [
        F("toolCallId", 4, type="string"),
    ],
    "McpToolsToolCallError": [
        F("error", 1, type="string"),
    ],
    "McpToolsToolCallResult": [
        F("success", 1, type="M:McpToolsToolCallSuccess"),
        F("error", 2, type="M:McpToolsToolCallError"),
    ],
    "McpToolsToolCallSuccess": [
        F("content", 1, type="string"),
    ],
    "Run_req": [],
    "GetDefaultModelForCli_req": [],
    "GetUsableModels_req": [],
    "TrackEvents_res": [],
    "GetEffectiveUserPlugins_req": [],
    "GetManagedSkills_req": [],
    "GetMe_req": [],
    "GetTeamAdminSettingsOrEmptyIfNotInTeam_req": [],
    "GetTeamAdminSettingsOrEmptyIfNotInTeam_res": [],
    "GetTeamReposOrEmptyIfNotInTeam_req": [],
    "GetTeamReposOrEmptyIfNotInTeam_res": [],
    "GetTeams_req": [],
    "GetTeams_res": [],
    "GetUserPrivacyMode_req": [],
    "ListMarketplaces_req": [],
    "ReportDistribution_res": [],
    "ReportIncrement_res": [],
    "GetServerConfig_req": [],
}

ENDPOINTS = {
    "agent.v1.AgentService.Run [res]": "Run_res",
    "aiserver.v1.AiService.AvailableModels [req]": "AvailableModels_req",
    "aiserver.v1.AiService.AvailableModels [res]": "AvailableModels_res",
    "aiserver.v1.AiService.GetDefaultModelForCli [res]": "GetDefaultModelForCli_res",
    "aiserver.v1.AiService.GetUsableModels [res]": "GetUsableModels_res",
    "aiserver.v1.AiService.NameAgent [req]": "NameAgent_req",
    "aiserver.v1.AiService.NameAgent [res]": "NameAgent_res",
    "aiserver.v1.AnalyticsService.SubmitLogs [req]": "SubmitLogs_req",
    "aiserver.v1.AnalyticsService.SubmitLogs [res]": "SubmitLogs_res",
    "aiserver.v1.AnalyticsService.TrackEvents [req]": "TrackEvents_req",
    "aiserver.v1.DashboardService.GetCliDownloadUrl [req]": "GetCliDownloadUrl_req",
    "aiserver.v1.DashboardService.GetCliDownloadUrl [res]": "GetCliDownloadUrl_res",
    "aiserver.v1.DashboardService.GetEffectiveUserPlugins [res]": "GetEffectiveUserPlugins_res",
    "aiserver.v1.DashboardService.GetGlobalCommands [req]": "GetGlobalCommands_req",
    "aiserver.v1.DashboardService.GetGlobalCommands [res]": "GetGlobalCommands_res",
    "aiserver.v1.DashboardService.GetManagedSkills [res]": "GetManagedSkills_res",
    "aiserver.v1.DashboardService.GetMe [res]": "GetMe_res",
    "aiserver.v1.DashboardService.GetUserPrivacyMode [res]": "GetUserPrivacyMode_res",
    "aiserver.v1.DashboardService.ListMarketplaces [res]": "ListMarketplaces_res",
    "aiserver.v1.MetricsService.ReportDistribution [req]": "ReportDistribution_req",
    "aiserver.v1.MetricsService.ReportIncrement [req]": "ReportIncrement_req",
    "aiserver.v1.ServerConfigService.GetServerConfig [res]": "GetServerConfig_res",
    "agent.v1.AgentService.Run [req]": "Run_req",
    "aiserver.v1.AiService.GetDefaultModelForCli [req]": "GetDefaultModelForCli_req",
    "aiserver.v1.AiService.GetUsableModels [req]": "GetUsableModels_req",
    "aiserver.v1.AnalyticsService.TrackEvents [res]": "TrackEvents_res",
    "aiserver.v1.DashboardService.GetEffectiveUserPlugins [req]": "GetEffectiveUserPlugins_req",
    "aiserver.v1.DashboardService.GetManagedSkills [req]": "GetManagedSkills_req",
    "aiserver.v1.DashboardService.GetMe [req]": "GetMe_req",
    "aiserver.v1.DashboardService.GetTeamAdminSettingsOrEmptyIfNotInTeam [req]": "GetTeamAdminSettingsOrEmptyIfNotInTeam_req",
    "aiserver.v1.DashboardService.GetTeamAdminSettingsOrEmptyIfNotInTeam [res]": "GetTeamAdminSettingsOrEmptyIfNotInTeam_res",
    "aiserver.v1.DashboardService.GetTeamReposOrEmptyIfNotInTeam [req]": "GetTeamReposOrEmptyIfNotInTeam_req",
    "aiserver.v1.DashboardService.GetTeamReposOrEmptyIfNotInTeam [res]": "GetTeamReposOrEmptyIfNotInTeam_res",
    "aiserver.v1.DashboardService.GetTeams [req]": "GetTeams_req",
    "aiserver.v1.DashboardService.GetTeams [res]": "GetTeams_res",
    "aiserver.v1.DashboardService.GetUserPrivacyMode [req]": "GetUserPrivacyMode_req",
    "aiserver.v1.DashboardService.ListMarketplaces [req]": "ListMarketplaces_req",
    "aiserver.v1.MetricsService.ReportDistribution [res]": "ReportDistribution_res",
    "aiserver.v1.MetricsService.ReportIncrement [res]": "ReportIncrement_res",
    "aiserver.v1.ServerConfigService.GetServerConfig [req]": "GetServerConfig_req",
}

UNKNOWN = "__unknown__"

_FIXED = {"double": ("d", 8), "fixed64": ("Q", 8),
          "float": ("f", 4), "fixed32": ("I", 4)}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7

def _enc_varint(v):
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        x = v & 0x7F; v >>= 7
        out.append(x | 0x80 if v else x)
        if not v:
            return bytes(out)

def _zz_enc(v): return (v << 1) ^ (v >> 63)
def _zz_dec(v): return (v >> 1) ^ -(v & 1)

def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            l, i = _varint(buf, i); v = buf[i:i + l]; i += l
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"bad wire type {wt}")
        yield fn, wt, v

def _wt_of(ftype):
    return {"bool": 0, "int32": 0, "int64": 0, "sint64": 0,
            "double": 1, "fixed64": 1, "float": 5, "fixed32": 5}.get(ftype, 2)

def _scalar(ftype, wt, raw):
    if wt == 0:
        if ftype == "sint64":
            return _zz_dec(raw)
        return bool(raw) if ftype == "bool" else raw
    if wt != 2:
        if ftype in ("double", "float"):
            return struct.unpack("<" + _FIXED[ftype][0], raw)[0]
        return int.from_bytes(raw, "little")
    return raw.decode("utf-8") if ftype == "string" else raw

def _enc_scalar(ftype, v):
    if ftype == "bool":
        return _enc_varint(int(bool(v)))
    if ftype in ("int32", "int64"):
        return _enc_varint(int(v))
    if ftype == "sint64":
        return _enc_varint(_zz_enc(int(v)))
    if ftype in _FIXED:
        num = float(v) if ftype in ("double", "float") else int(v)
        return struct.pack("<" + _FIXED[ftype][0], num)
    b = v.encode("utf-8") if ftype == "string" else v
    return _enc_varint(len(b)) + b


def decode(data, msg_type):
    by_num = {f.num: f for f in MESSAGES[msg_type]}
    out, unk = {}, {}
    for fn, wt, raw in _fields(bytes(data)):
        f = by_num.get(fn)
        if f is None:
            unk.setdefault(str(fn), []).append((wt, raw))
            continue
        val = _decode_field(f, wt, raw)
        if f.enum is not None:
            if isinstance(val, list):
                val = [f.enum.get(item, item) for item in val]
            else:
                val = f.enum.get(val, val)
        if f.map:
            out.setdefault(f.name, []).append(val)
        elif f.packed and wt == 2:
            prev = out.get(f.name)
            if prev is None:
                out[f.name] = val
            elif prev and isinstance(prev[0], list):
                prev.append(val)
            else:
                out[f.name] = [prev, val]
        elif f.rep:
            out.setdefault(f.name, []).append(val)
        elif f.name in out:
            if not isinstance(out[f.name], list):
                out[f.name] = [out[f.name]]
            out[f.name].append(val)
        else:
            out[f.name] = val
    if unk:
        out[UNKNOWN] = unk
    return out

def _decode_field(f, wt, raw):
    if f.map:
        k, vals = None, []
        for efn, ewt, ev in _fields(raw):
            if efn == 1:
                k = _scalar(f.map[0], ewt, ev)
            elif efn == 2:
                vt = f.map[1]
                vals.append(decode(ev, vt[2:]) if vt.startswith("M:")
                            else _scalar(vt, ewt, ev))
        return (k, vals[0] if len(vals) == 1 else (vals or None))
    if f.type and f.type.startswith("M:"):
        return decode(raw, f.type[2:])
    if f.packed and wt == 2:
        if f.type in ("int32", "int64", "sint64"):
            vals, i = [], 0
            while i < len(raw):
                v, i = _varint(raw, i)
                vals.append(_zz_dec(v) if f.type == "sint64" else v)
            return vals
        ch, size = _FIXED[f.type]
        return list(struct.unpack(f"<{len(raw) // size}{ch}", raw))
    return _scalar(f.type, wt, raw)

def _unlab(f, v):
    if isinstance(v, list):
        return [_unlab(f, item) for item in v]
    if isinstance(v, str) and getattr(f, "enum", None):
        for iv, lab in f.enum.items():
            if lab == v:
                return iv
        if v.isdigit():
            return int(v)
        raise ValueError(f"unknown enum label {v!r} for {f.name} "
                         f"(known: {sorted(f.enum.values())})")
    return v


def encode(obj, msg_type):
    msgs = MESSAGES[msg_type]
    by_name = {g.name: g for g in msgs}
    by_name.update({f"f{g.num}": g for g in msgs})
    items, seen = [], {}
    for k, v in obj.items():
        if v is None:
            continue
        if k == UNKNOWN:
            for num, occs in v.items():
                n = int(num)
                if n in seen and seen[n] != UNKNOWN:
                    raise ValueError(f"field {n} of {msg_type} supplied "
                                     f"twice: {seen[n]!r} and {UNKNOWN}")
                seen[n] = UNKNOWN
                for wt, raw in occs:
                    items.append((n, None,
                                  (wt, _enc_varint(raw) if isinstance(raw, int)
                                   else raw)))
            continue
        f = by_name.get(k)
        if f is None:
            valid = sorted({g.name for g in msgs}
                           | {f"f{g.num}" for g in msgs} | {UNKNOWN})
            raise KeyError(f"{k!r} is not a field of {msg_type}; "
                           f"valid keys: {valid}")
        if f.num in seen:
            raise ValueError(f"field {f.num} of {msg_type} supplied twice: "
                             f"{seen[f.num]!r} and {k!r}")
        seen[f.num] = k
        els = [v] if f.packed else (v if isinstance(v, list) else [v])
        for el in els:
            items.append((f.num, f, el if f.map else _unlab(f, el)))
    items.sort(key=lambda t: t[0])
    parts = []
    for fn, f, el in items:
        if f is None:
            wt, raw = el
            parts.append(_enc_varint(fn << 3 | wt))
            if wt == 2:
                parts.append(_enc_varint(len(raw)))
            parts.append(raw)
        else:
            parts.append(_encode_field(fn, f, el))
    return b"".join(parts)

def _encode_field(fn, f, v):
    if f.map:
        kt, vt = f.map
        inner = []
        if v[0] is not None:
            inner.append((1, kt, v[0]))
        if v[1] is not None:
            inner.extend((2, vt, e) for e in
                         (v[1] if isinstance(v[1], list) else [v[1]]))
        blob = b"".join(_emit(n, t, e) for n, t, e in inner)
        return _enc_varint(fn << 3 | 2) + _enc_varint(len(blob)) + blob
    if f.packed:
        runs = v if (v and isinstance(v[0], list)) else [v]
        return b"".join(
            _enc_varint(fn << 3 | 2) + _enc_varint(len(p)) + p
            for p in (b"".join(_enc_scalar(f.type, e) for e in run)
                      for run in runs))
    return _emit(fn, f.type, v)

def _emit(num, ftype, v):
    if ftype.startswith("M:"):
        blob = encode(v, ftype[2:])
        return _enc_varint(num << 3 | 2) + _enc_varint(len(blob)) + blob
    return _enc_varint(num << 3 | _wt_of(ftype)) + _enc_scalar(ftype, v)
