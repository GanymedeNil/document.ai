// 对外接口的数据结构，依据 eaer/api/openapi.yaml（接口的唯一定义）。改接口时先改 openapi.yaml，再同步这里。

export type Mode = "fast" | "deep" | "verified";
export type ValidGrade = "explicit" | "inferred" | "unknown";
export type DocType = "normative" | "record";
export type Locator = Record<string, unknown>;

export interface ApiErrorBody {
  code: string;
  message: string;
  request_id: string;
}

export interface QueryRequest {
  question: string;
  session_id?: string | null;
  as_of_valid?: string | null;
  as_of_system?: string | null;
  mode_hint?: Mode | null;
  /** 严格模式（true）或宽松模式（false）；null 或不传时取租户默认 */
  strict?: boolean | null;
  stream?: boolean;
}

/** 租户默认的回答方式（GET/PUT /v1/config/answer-policy） */
export interface AnswerPolicy {
  strict: boolean;
}

/** 结论的依据：证据原文；按证据里的单价写出算式算出；宽松模式下的常识推断 */
export type ClaimBasis = "evidence" | "calculation" | "common_knowledge";

export interface TimeBasis {
  as_of_valid: string | null;
  as_of_system: string | null;
  source: "parameter" | "parsed" | "model" | "today";
  unresolved?: string[];
}

export interface Citation {
  evidence_id: string;
  document_id: string;
  version_id: string;
  version_no: number | null;
  locator: Locator;
  valid_grade: ValidGrade | null;
}

export interface Claim {
  text: string;
  status: "supported" | "unverified";
  core: boolean;
  basis: ClaimBasis;
  citations: Citation[];
}

/** 模型自身的思考文字（生成服务开启思考时），按编排步骤。 */
export interface ReasoningStep {
  step: "prepare" | "first" | "escalate" | "verify" | "assemble";
  text: string;
}

export interface QueryResponse {
  session_id: string;
  status: "completed" | "running" | "cancelled";
  mode: Mode | null;
  mode_path: Mode[];
  time_basis: TimeBasis | null;
  answer: string | null;
  claims: Claim[];
  abstained: boolean | null;
  notices: string[];
  reasoning: ReasoningStep[];
  /** 这次用的回答方式；转入后台或取消时为 null */
  strict: boolean | null;
}

export interface Progress {
  done: string[];
  next: string | null;
  mode_path: Mode[];
  retrieval_rounds: number;
  retry_after?: string;
  waiting?: string | null;
}

export interface SessionItem {
  query_id: string;
  asked_at: string;
  question: string;
  time_basis: TimeBasis | Record<string, never> | null;
  mode: Mode;
  mode_path: Mode[];
  status: "running" | "completed" | "cancelled" | "failed";
  answer: string | null;
  abstained: boolean;
  background: boolean;
  claims: Claim[];
  reasoning: ReasoningStep[];
  /** 这次用的回答方式（true 严格、false 宽松）；加入这个开关之前的提问为 null */
  strict: boolean | null;
  progress?: Progress;
}

export interface Session {
  session_id: string;
  items: SessionItem[];
}

export interface CancelResult {
  session_id: string;
  items: { query_id: string; status: "cancelled" | "cancelling" }[];
}

export interface Validity {
  valid_from: string | null;
  valid_from_grade: ValidGrade;
  valid_to: string | null;
  valid_to_grade: ValidGrade;
}

export interface SearchableEvidence {
  evidence_id: string;
  status: "searchable";
  modality: "text" | "table" | "figure";
  content: string;
  locator: Locator;
  parse_stage: "text" | "deep";
  document: { id: string; title: string; doc_type: DocType | null };
  version: { id: string; version_no: number; approved_at: string; doc_date: string | null; publisher: string | null };
  validity: Validity | null;
}

export interface ArchivedEvidence {
  evidence_id: string;
  status: "archived";
  notice: string;
  document: { id: string; title: string };
  version: { id: string; version_no: number };
}

export type Evidence = SearchableEvidence | ArchivedEvidence;

export interface UploadResult {
  document_id: string;
  version_id: string;
  status: "pending_review";
  duplicate: boolean;
}

export interface ReviewTask {
  task_id: string;
  batch_id: string | null;
  status: "open" | "approved" | "deleted";
  suggestions: Record<string, unknown> | null;
  corrections: Record<string, unknown> | null;
  reviewer: string | null;
  decided_at: string | null;
  version_id: string | null;
  version_no: number | null;
  uploaded_at: string | null;
  import_batch_id: string | null;
  document_id: string | null;
  title: string | null;
  doc_type: DocType | null;
  extraction_status: "queued" | "running" | "done" | "failed" | null;
  draft_validity: Record<string, unknown> | null;
}

export interface ReviewTaskPage {
  items: ReviewTask[];
  next_cursor: string | null;
}

export interface DatedValue {
  value: string;
  grade?: ValidGrade;
}

export interface RelationInput {
  relation?: "supersedes" | "conflicts" | "references";
  to_document_id: string;
  to_version_no: number;
}

export interface Corrections {
  doc_type?: DocType;
  valid_from?: DatedValue | null;
  valid_to?: DatedValue | null;
  relations?: RelationInput[] | null;
}

export interface Decision {
  task_id: string;
  decision: "approve" | "delete";
  corrections?: Corrections | null;
}

export interface DecisionResult {
  task_id: string;
  ok: boolean;
  error: string | null;
}

export interface ArchiveResult {
  document_id: string;
  versions: string[];
  status: "archived" | "searchable";
}

export interface ForcedVerifyRule {
  id: string;
  kind: "keyword" | "document" | "doc_type" | "all";
  value: string | null;
  description: string | null;
  created_by: string;
  created_at: string;
}

export interface ForcedVerifyList {
  builtin: { id: string; kind: "builtin"; description: string }[];
  rules: ForcedVerifyRule[];
}

export interface VersionEvidence {
  evidence_id: string;
  modality: "text" | "table" | "figure";
  content: string;
  locator: Locator;
  parse_stage: "text" | "deep";
}

export interface VersionContent {
  status: "searchable" | "pending_review";
  access: "permission" | "review";
  document: { id: string; title: string; doc_type: DocType | null };
  version: {
    id: string; version_no: number; uploaded_at: string; approved_at: string | null;
    doc_date: string | null; publisher: string | null;
  };
  validity: Validity | null;
  original: { filename: string; content_type: string };
  evidence: VersionEvidence[];
}

export interface ArchivedVersion {
  status: "archived";
  notice: string;
  document: { id: string; title: string };
  version: { id: string; version_no: number };
}

export type VersionView = VersionContent | ArchivedVersion;
