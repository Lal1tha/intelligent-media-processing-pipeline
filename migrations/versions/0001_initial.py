"""initial schema"""
from alembic import op
import sqlalchemy as sa
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None
def upgrade():
    op.create_table("uploads", sa.Column("id", sa.String(36), primary_key=True), sa.Column("original_filename", sa.String(255), nullable=False), sa.Column("stored_filename", sa.String(255), nullable=False, unique=True), sa.Column("file_path", sa.String(1024), nullable=False), sa.Column("mime_type", sa.String(100), nullable=False), sa.Column("file_size", sa.Integer, nullable=False), sa.Column("sha256", sa.String(64), nullable=False), sa.Column("perceptual_hash", sa.String(32), nullable=False), sa.Column("status", sa.Enum("pending","processing","completed","failed", name="processingstatus"), nullable=False), sa.Column("failure_reason", sa.Text), sa.Column("uploaded_at", sa.DateTime, nullable=False), sa.Column("created_at", sa.DateTime, nullable=False), sa.Column("updated_at", sa.DateTime, nullable=False), sa.Column("processed_at", sa.DateTime))
    op.create_index("ix_uploads_sha256", "uploads", ["sha256"]); op.create_index("ix_uploads_perceptual_hash", "uploads", ["perceptual_hash"]); op.create_index("ix_uploads_status", "uploads", ["status"])
    op.create_table("analysis_results", sa.Column("id", sa.Integer, primary_key=True), sa.Column("processing_id", sa.String(36), sa.ForeignKey("uploads.id"), unique=True, nullable=False), sa.Column("overall_status", sa.String(40), nullable=False), sa.Column("result_json", sa.JSON, nullable=False), sa.Column("created_at", sa.DateTime, nullable=False))
def downgrade():
    op.drop_table("analysis_results"); op.drop_table("uploads")
