"""episodic memory tables

Revision ID: 0036426a99ae
Revises: 77cb94e364ce
Create Date: 2026-07-12 11:48:44.125558

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0036426a99ae'
down_revision: str | None = '77cb94e364ce'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Episodic memory pipeline (no pgvector columns, so no extension needed here).
    # Raw per-turn observations; conversation_id is intentionally NOT a FK — there
    # is no conversations table (chat stays otherwise stateless server-side).
    op.create_table(
        'memory_observations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('topic_key', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_memory_observations_user_id'), 'memory_observations', ['user_id'], unique=False
    )
    op.create_index(
        'ix_memory_observations_user_topic',
        'memory_observations',
        ['user_id', 'topic_key'],
        unique=False,
    )

    # Consolidated, durable memories the agent reads every turn. Exactly one row per
    # (user_id, topic_key) — the unique constraint backs the consolidation upsert.
    op.create_table(
        'user_memories',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('topic_key', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('source_chat_count', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'topic_key', name='uq_user_memories_user_topic'),
    )


def downgrade() -> None:
    op.drop_table('user_memories')
    op.drop_index('ix_memory_observations_user_topic', table_name='memory_observations')
    op.drop_index(op.f('ix_memory_observations_user_id'), table_name='memory_observations')
    op.drop_table('memory_observations')
