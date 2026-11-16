"""审核任务记录所属租户（开发文档 6.3、11.2）。

删除待审核版本后，审核任务保留谁在何时删除，但不再指向版本（deleted_has_no_content）。此前租户只能经
“任务 → 版本 → 文档”得知，删除后这条路断了，按租户列出 status=deleted 的任务因此永远为空。
本迁移给 review_task 加 tenant_id：
- 回填：指向版本的任务取版本所属文档的租户。迁移前已删除的任务无从得知租户，不推断，保持为空；
  它们不会出现在任何租户的列表里（迁移前本来也列不出来）。
- 此后写入或修改的任务必须带租户：检查约束用 NOT VALID，只约束新写入与修改的行，不追溯上面那些无从回填的旧行。
- 触发器（第二道防线）：任务指向版本时，租户必须与版本所属文档的租户相同；租户写入后不可修改。
- 按租户、状态列出任务的索引（排序与游标按 id 的文本，与 Review.list_tasks 一致）。

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-14
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(r"""
        ALTER TABLE review_task ADD COLUMN tenant_id text;

        UPDATE review_task t SET tenant_id = d.tenant_id
          FROM document_version v JOIN document d ON d.id = v.document_id
         WHERE v.id = t.version_id;

        ALTER TABLE review_task ADD CONSTRAINT review_task_tenant_required CHECK (tenant_id IS NOT NULL) NOT VALID;

        CREATE INDEX review_task_by_tenant ON review_task (tenant_id, status, (id::text));

        CREATE FUNCTION eaer_guard_review_task_tenant() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE
          doc_tenant text;
        BEGIN
          IF TG_OP = 'UPDATE' AND OLD.tenant_id IS DISTINCT FROM NEW.tenant_id THEN
            RAISE EXCEPTION 'invariant: 审核任务的租户写入后不可修改（%）', OLD.id;
          END IF;
          IF NEW.version_id IS NOT NULL THEN
            SELECT d.tenant_id INTO doc_tenant
              FROM document_version v JOIN document d ON d.id = v.document_id
             WHERE v.id = NEW.version_id;
            IF doc_tenant IS DISTINCT FROM NEW.tenant_id THEN
              RAISE EXCEPTION 'invariant: 审核任务的租户（%）与版本所属文档的租户（%）不同', NEW.tenant_id, doc_tenant;
            END IF;
          END IF;
          RETURN NEW;
        END $$;
        CREATE TRIGGER review_task_tenant_guard BEFORE INSERT OR UPDATE ON review_task
          FOR EACH ROW EXECUTE FUNCTION eaer_guard_review_task_tenant();
    """)


def downgrade() -> None:
    op.execute(r"""
        DROP TRIGGER IF EXISTS review_task_tenant_guard ON review_task;
        DROP FUNCTION IF EXISTS eaer_guard_review_task_tenant();
        DROP INDEX IF EXISTS review_task_by_tenant;
        ALTER TABLE review_task DROP CONSTRAINT IF EXISTS review_task_tenant_required;
        ALTER TABLE review_task DROP COLUMN IF EXISTS tenant_id;
    """)
