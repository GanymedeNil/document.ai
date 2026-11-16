"""初始数据模型：开发文档第 4 节的 11 张表与不变量。

Revision ID: 0001
Revises:
Create Date: 2026-11-11

不变量由数据库直接保证（触发器、检查约束、部分唯一索引），
应用代码出错时数据库拒绝写入，而不是事后对账才发现。
触发器报错的信息以“invariant:”开头，便于测试和日志检索。
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = """
CREATE TABLE tenant_config (
  tenant_id            text PRIMARY KEY,
  forced_verify_rules  jsonb NOT NULL DEFAULT '[]',
  thresholds           jsonb NOT NULL DEFAULT '{}',
  budgets              jsonb NOT NULL DEFAULT '{}',
  language_config      jsonb NOT NULL DEFAULT '{}',
  trace_retention_days int   NOT NULL DEFAULT 30 CHECK (trace_retention_days > 0),
  updated_at           timestamptz NOT NULL
);

CREATE TABLE acl_policy (
  id          uuid PRIMARY KEY,
  tenant_id   text NOT NULL REFERENCES tenant_config(tenant_id),
  principals  text[] NOT NULL CHECK (cardinality(principals) > 0),
  source      text NOT NULL CHECK (source IN ('reference', 'connector')),
  updated_at  timestamptz NOT NULL
);

CREATE TABLE document (
  id          uuid PRIMARY KEY,
  tenant_id   text NOT NULL REFERENCES tenant_config(tenant_id),
  source_ref  text NOT NULL,
  title       text NOT NULL,
  doc_type    text CHECK (doc_type IN ('normative', 'record')),
  language    text,
  created_at  timestamptz NOT NULL,
  UNIQUE (tenant_id, source_ref)
);

CREATE TABLE document_version (
  id                uuid PRIMARY KEY,
  document_id       uuid NOT NULL REFERENCES document(id),
  version_no        int  NOT NULL CHECK (version_no > 0),
  status            text NOT NULL CHECK (status IN ('pending_review', 'searchable', 'archived')),
  content_hash      text NOT NULL,
  raw_object_uri    text NOT NULL,
  acl_policy_id     uuid NOT NULL REFERENCES acl_policy(id),
  uploaded_at       timestamptz NOT NULL,
  source_created_at timestamptz,
  import_batch_id   text,
  approved_at       timestamptz,
  archived_at       timestamptz,
  doc_date          date,
  parser_version    text NOT NULL,
  UNIQUE (document_id, version_no),
  -- 只有待审核的版本没有审核通过时间
  CONSTRAINT approved_iff_published CHECK ((status = 'pending_review') = (approved_at IS NULL)),
  -- 归档状态与归档时间一致
  CONSTRAINT archived_iff_archived_at CHECK ((status = 'archived') = (archived_at IS NOT NULL)),
  CONSTRAINT approved_after_upload CHECK (approved_at IS NULL OR approved_at >= uploaded_at)
);

CREATE TABLE version_validity (
  id               uuid PRIMARY KEY,
  version_id       uuid NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
  valid_from       date,
  valid_from_grade text NOT NULL CHECK (valid_from_grade IN ('explicit', 'inferred', 'unknown')),
  valid_to         date,
  valid_to_grade   text NOT NULL CHECK (valid_to_grade IN ('explicit', 'inferred', 'unknown')),
  valid_to_reason  text CHECK (valid_to_reason IN ('new_version', 'superseded', 'stated')),
  unresolved_text  text,
  locator          jsonb,
  recorded_at      timestamptz,
  replaced_at      timestamptz,
  CONSTRAINT replaced_needs_recorded CHECK (replaced_at IS NULL OR recorded_at IS NOT NULL),
  CONSTRAINT replaced_not_before_recorded CHECK (replaced_at IS NULL OR replaced_at >= recorded_at),
  CONSTRAINT interval_ordered CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_to > valid_from)
);
-- 每个版本同一时刻只有一行当前有效期
CREATE UNIQUE INDEX version_validity_one_current
  ON version_validity (version_id) WHERE recorded_at IS NOT NULL AND replaced_at IS NULL;
-- 每个版本至多一行审核前的待确认值
CREATE UNIQUE INDEX version_validity_one_draft
  ON version_validity (version_id) WHERE recorded_at IS NULL;

CREATE TABLE version_relation (
  id              uuid PRIMARY KEY,
  relation        text NOT NULL CHECK (relation IN ('supersedes', 'conflicts', 'references')),
  from_version_id uuid NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
  to_version_id   uuid NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
  grade           text NOT NULL CHECK (grade IN ('explicit', 'inferred')),
  locator         jsonb,
  recorded_at     timestamptz,
  retracted_at    timestamptz,
  CONSTRAINT retracted_needs_recorded CHECK (retracted_at IS NULL OR recorded_at IS NOT NULL),
  CONSTRAINT retracted_not_before_recorded CHECK (retracted_at IS NULL OR retracted_at >= recorded_at),
  CONSTRAINT no_self_relation CHECK (from_version_id <> to_version_id)
);
CREATE INDEX version_relation_from ON version_relation (from_version_id);
CREATE INDEX version_relation_to ON version_relation (to_version_id);

CREATE TABLE evidence (
  id            text PRIMARY KEY CHECK (id ~ '^ev_[0-9a-f]{16}$'),
  version_id    uuid NOT NULL REFERENCES document_version(id) ON DELETE CASCADE,
  modality      text NOT NULL CHECK (modality IN ('text', 'table', 'figure')),
  locator       jsonb NOT NULL,
  content       text NOT NULL,
  content_hash  text NOT NULL,
  parse_stage   text NOT NULL CHECK (parse_stage IN ('text', 'deep')),
  created_at    timestamptz NOT NULL
);
CREATE INDEX evidence_version ON evidence (version_id);

CREATE TABLE lifecycle_event (
  id          uuid PRIMARY KEY,
  version_id  uuid NOT NULL REFERENCES document_version(id),
  event       text NOT NULL CHECK (event IN
                ('published', 'archived', 'unarchived', 'acl_changed', 'validity_corrected')),
  actor       text NOT NULL,
  detail      jsonb,
  occurred_at timestamptz NOT NULL
);
CREATE INDEX lifecycle_event_version ON lifecycle_event (version_id, occurred_at);

CREATE TABLE review_task (
  id          uuid PRIMARY KEY,
  -- 删除待审核版本后保留审核记录（谁在何时删除），但不再指向版本、不保留抽取内容
  version_id  uuid REFERENCES document_version(id) ON DELETE SET NULL,
  batch_id    uuid,
  status      text NOT NULL CHECK (status IN ('open', 'approved', 'deleted')),
  corrections jsonb,
  reviewer    text,
  decided_at  timestamptz,
  CONSTRAINT open_has_version CHECK (status <> 'open' OR version_id IS NOT NULL),
  CONSTRAINT deleted_has_no_content CHECK (status <> 'deleted' OR (version_id IS NULL AND corrections IS NULL))
);
CREATE INDEX review_task_open ON review_task (batch_id) WHERE status = 'open';

CREATE TABLE ingest_job (
  id           uuid PRIMARY KEY,
  version_id   uuid REFERENCES document_version(id) ON DELETE CASCADE,
  stage        text NOT NULL CHECK (stage IN ('receive', 'parse_text', 'classify_extract',
                 'publish_index', 'parse_deep', 'index_update', 'reindex')),
  status       text NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
  priority     int  NOT NULL DEFAULT 100,
  attempts     int  NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  last_error   text,
  locked_until timestamptz,
  created_at   timestamptz NOT NULL,
  updated_at   timestamptz NOT NULL
);
CREATE INDEX ingest_job_queue ON ingest_job (priority, created_at) WHERE status = 'queued';

CREATE TABLE query_log (
  id             uuid PRIMARY KEY,
  tenant_id      text NOT NULL,
  session_id     uuid NOT NULL,
  user_id        text NOT NULL,
  asked_at       timestamptz NOT NULL,
  question       text NOT NULL,
  time_basis     jsonb,
  mode           text NOT NULL CHECK (mode IN ('fast', 'deep', 'verified')),
  mode_path      text[] NOT NULL CHECK (cardinality(mode_path) > 0),
  status         text NOT NULL CHECK (status IN ('running', 'completed', 'cancelled', 'failed')),
  answer         text,
  claims         jsonb NOT NULL DEFAULT '[]',
  evidence_ids   text[] NOT NULL DEFAULT '{}',
  abstained      boolean NOT NULL DEFAULT false,
  authz_decision jsonb NOT NULL,
  model_id       text NOT NULL,
  index_build_id text NOT NULL,
  cost           jsonb
);
CREATE INDEX query_log_session ON query_log (session_id, asked_at);
"""

TRIGGERS = r"""
-- 版本：只有待审核可删除；approved_at 只写一次；状态只能按允许的路径转换；发布后身份字段不可改
CREATE FUNCTION eaer_guard_version_delete() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status <> 'pending_review' THEN
    RAISE EXCEPTION 'invariant: 只有待审核的版本可以删除（版本 %，状态 %）', OLD.id, OLD.status;
  END IF;
  RETURN OLD;
END $$;
CREATE TRIGGER version_delete_guard BEFORE DELETE ON document_version
  FOR EACH ROW EXECUTE FUNCTION eaer_guard_version_delete();

CREATE FUNCTION eaer_guard_version_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.approved_at IS NOT NULL AND NEW.approved_at IS DISTINCT FROM OLD.approved_at THEN
    RAISE EXCEPTION 'invariant: approved_at 只能写一次（版本 %）', OLD.id;
  END IF;
  IF NOT ((OLD.status, NEW.status) IN (
      ('pending_review', 'pending_review'), ('pending_review', 'searchable'),
      ('searchable', 'searchable'), ('searchable', 'archived'),
      ('archived', 'archived'), ('archived', 'searchable'))) THEN
    RAISE EXCEPTION 'invariant: 不允许的状态转换 % -> %（版本 %）', OLD.status, NEW.status, OLD.id;
  END IF;
  IF OLD.status <> 'pending_review' AND (
       NEW.document_id <> OLD.document_id OR NEW.version_no <> OLD.version_no
    OR NEW.content_hash <> OLD.content_hash OR NEW.raw_object_uri <> OLD.raw_object_uri
    OR NEW.uploaded_at <> OLD.uploaded_at) THEN
    RAISE EXCEPTION 'invariant: 已发布版本的内容与身份字段不可修改（版本 %）', OLD.id;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER version_update_guard BEFORE UPDATE ON document_version
  FOR EACH ROW EXECUTE FUNCTION eaer_guard_version_update();

-- 有效期：已记录的行只能写入 replaced_at，不能修改或删除
CREATE FUNCTION eaer_guard_validity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.recorded_at IS NULL THEN
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
  END IF;
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'invariant: 已记录的有效期行不可删除（%）', OLD.id;
  END IF;
  IF OLD.replaced_at IS NULL AND NEW.replaced_at IS NOT NULL
     AND (NEW.version_id, NEW.valid_from, NEW.valid_from_grade, NEW.valid_to, NEW.valid_to_grade,
          NEW.valid_to_reason, NEW.unresolved_text, NEW.locator, NEW.recorded_at)
         IS NOT DISTINCT FROM
         (OLD.version_id, OLD.valid_from, OLD.valid_from_grade, OLD.valid_to, OLD.valid_to_grade,
          OLD.valid_to_reason, OLD.unresolved_text, OLD.locator, OLD.recorded_at) THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'invariant: 已记录的有效期行只能写入 replaced_at（%）', OLD.id;
END $$;
CREATE TRIGGER validity_guard BEFORE UPDATE OR DELETE ON version_validity
  FOR EACH ROW EXECUTE FUNCTION eaer_guard_validity();

-- 关系：已记录的行只能写入 retracted_at
CREATE FUNCTION eaer_guard_relation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.recorded_at IS NULL THEN
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
  END IF;
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'invariant: 已记录的关系行不可删除（%）', OLD.id;
  END IF;
  IF OLD.retracted_at IS NULL AND NEW.retracted_at IS NOT NULL
     AND (NEW.relation, NEW.from_version_id, NEW.to_version_id, NEW.grade, NEW.locator, NEW.recorded_at)
         IS NOT DISTINCT FROM
         (OLD.relation, OLD.from_version_id, OLD.to_version_id, OLD.grade, OLD.locator, OLD.recorded_at) THEN
    RETURN NEW;
  END IF;
  RAISE EXCEPTION 'invariant: 已记录的关系行只能写入 retracted_at（%）', OLD.id;
END $$;
CREATE TRIGGER relation_guard BEFORE UPDATE OR DELETE ON version_relation
  FOR EACH ROW EXECUTE FUNCTION eaer_guard_relation();

-- 生命周期事件只追加
CREATE FUNCTION eaer_guard_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'invariant: % 只能追加，不能修改或删除', TG_TABLE_NAME;
END $$;
CREATE TRIGGER lifecycle_event_append_only BEFORE UPDATE OR DELETE ON lifecycle_event
  FOR EACH ROW EXECUTE FUNCTION eaer_guard_append_only();
"""

DROP = """
DROP TABLE IF EXISTS query_log, ingest_job, review_task, lifecycle_event, evidence,
  version_relation, version_validity, document_version, document, acl_policy, tenant_config CASCADE;
DROP FUNCTION IF EXISTS eaer_guard_version_delete(), eaer_guard_version_update(),
  eaer_guard_validity(), eaer_guard_relation(), eaer_guard_append_only();
"""


def upgrade() -> None:
    op.execute(TABLES)
    op.execute(TRIGGERS)


def downgrade() -> None:
    op.execute(DROP)
