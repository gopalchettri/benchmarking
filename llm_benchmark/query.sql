USE [netra-ai-rag-db]
GO
/****** Object:  Table [dbo].[netra_chatbot_audit_log]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_audit_log](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[user_id] [uniqueidentifier] NULL,
	[scope_entity_id] [uniqueidentifier] NULL,
	[session_id] [nvarchar](128) NULL,
	[correlation_id] [nvarchar](128) NULL,
	[ip_address] [nvarchar](45) NULL,
	[user_agent] [nvarchar](512) NULL,
	[action_category] [nvarchar](128) NOT NULL,
	[action_type] [nvarchar](128) NOT NULL,
	[entity_type] [nvarchar](128) NULL,
	[resource_id] [nvarchar](128) NULL,
	[metadata_json] [nvarchar](max) NULL,
	[created_at] [datetime2](7) NOT NULL,
	[before_state] [nvarchar](max) NULL,
	[after_state] [nvarchar](max) NULL,
	[previous_checksum] [nvarchar](128) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_benchmark_results]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_benchmark_results](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [nvarchar](128) NOT NULL,
	[suite_name] [nvarchar](256) NOT NULL,
	[retrieval_recall_at_k] [float] NOT NULL,
	[grounding_precision] [float] NOT NULL,
	[language_parity_ratio] [float] NOT NULL,
	[security_pass_rate] [float] NOT NULL,
	[kgrag_coverage_ratio] [float] NOT NULL,
	[total_queries] [int] NOT NULL,
	[passed] [bit] NOT NULL,
	[failures] [nvarchar](max) NULL,
	[created_at] [datetimeoffset](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_chunk_concept_mapping]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_chunk_concept_mapping](
	[id] [uniqueidentifier] NOT NULL,
	[chunk_id] [uniqueidentifier] NOT NULL,
	[concept_id] [uniqueidentifier] NOT NULL,
	[confidence] [decimal](10, 4) NULL,
	[mapping_source] [nvarchar](64) NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[creation_date] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_chunk_concept_mapping] UNIQUE NONCLUSTERED 
(
	[tenant_id] ASC,
	[chunk_id] ASC,
	[concept_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_conversation_feedback]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_conversation_feedback](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[user_id] [uniqueidentifier] NOT NULL,
	[conversation_id] [uniqueidentifier] NOT NULL,
	[message_id] [uniqueidentifier] NOT NULL,
	[rating] [int] NOT NULL,
	[comment] [nvarchar](max) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_conversation_message]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_conversation_message](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[session_id] [uniqueidentifier] NOT NULL,
	[role] [nvarchar](64) NOT NULL,
	[content] [nvarchar](max) NOT NULL,
	[grounded] [bit] NULL,
	[pipeline_metadata] [nvarchar](max) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_conversation_session]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_conversation_session](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[user_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[title] [nvarchar](512) NULL,
	[status] [nvarchar](64) NOT NULL,
	[turn_count] [int] NOT NULL,
	[first_query] [nvarchar](max) NULL,
	[last_intent] [nvarchar](128) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
	[updated_at] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_escalation_requests]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_escalation_requests](
	[id] [uniqueidentifier] NOT NULL,
	[escalation_id] [uniqueidentifier] NOT NULL,
	[tenant_id] [nvarchar](256) NOT NULL,
	[query] [nvarchar](max) NOT NULL,
	[reason] [nvarchar](max) NOT NULL,
	[planner_confidence] [float] NOT NULL,
	[policy_sensitive] [bit] NOT NULL,
	[status] [nvarchar](64) NOT NULL,
	[reviewer_response] [nvarchar](max) NULL,
	[responder] [nvarchar](256) NULL,
	[responded_at] [datetime2](7) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_escalation_requests] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_escalation_escalation_id] UNIQUE NONCLUSTERED 
(
	[escalation_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_experiments]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_experiments](
	[id] [uniqueidentifier] NOT NULL,
	[experiment_id] [uniqueidentifier] NOT NULL,
	[tenant_id] [nvarchar](256) NOT NULL,
	[name] [nvarchar](256) NOT NULL,
	[variant_a] [nvarchar](256) NOT NULL,
	[variant_b] [nvarchar](256) NOT NULL,
	[traffic_split] [float] NOT NULL,
	[status] [nvarchar](64) NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_experiments] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_experiments_experiment_id] UNIQUE NONCLUSTERED 
(
	[experiment_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_idempotency_results]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_idempotency_results](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[subject_id] [uniqueidentifier] NOT NULL,
	[command_name] [nvarchar](128) NOT NULL,
	[idempotency_key] [nvarchar](256) NOT NULL,
	[request_hash] [nvarchar](128) NOT NULL,
	[response_body] [nvarchar](max) NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_idempotency_results] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_ingestion_run]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_ingestion_run](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[run_type] [nvarchar](64) NOT NULL,
	[source_module] [nvarchar](100) NULL,
	[status] [nvarchar](64) NOT NULL,
	[records_processed] [int] NOT NULL,
	[records_failed] [int] NOT NULL,
	[chunks_created] [int] NOT NULL,
	[embeddings_generated] [int] NOT NULL,
	[graph_nodes_synced] [int] NOT NULL,
	[parser_version] [nvarchar](100) NULL,
	[ontology_version] [nvarchar](100) NULL,
	[embedding_model] [nvarchar](100) NULL,
	[errors] [nvarchar](max) NULL,
	[created_by] [nvarchar](128) NULL,
	[started_at] [datetime2](7) NOT NULL,
	[completed_at] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_knowledge_chunk]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_knowledge_chunk](
	[id] [uniqueidentifier] NOT NULL,
	[chunk_id] [nvarchar](200) NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[document_id] [uniqueidentifier] NULL,
	[document_version_id] [uniqueidentifier] NULL,
	[title] [nvarchar](500) NULL,
	[content] [nvarchar](max) NOT NULL,
	[module_scope] [nvarchar](100) NULL,
	[resource_scope] [nvarchar](100) NULL,
	[section_path] [nvarchar](500) NULL,
	[chunk_sequence] [int] NULL,
	[source_type] [nvarchar](64) NULL,
	[source_record_id] [nvarchar](128) NULL,
	[ontology_concept_ids] [nvarchar](max) NULL,
	[access_roles] [nvarchar](max) NULL,
	[metadata] [nvarchar](max) NULL,
	[embedding_status] [nvarchar](64) NOT NULL,
	[embedding_model] [nvarchar](100) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_by] [nvarchar](128) NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_by] [nvarchar](128) NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_chunk_tenant_chunk_id] UNIQUE NONCLUSTERED 
(
	[tenant_id] ASC,
	[chunk_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_knowledge_document]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_knowledge_document](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[module] [nvarchar](100) NOT NULL,
	[resource] [nvarchar](100) NOT NULL,
	[source_type] [nvarchar](64) NOT NULL,
	[source_system] [nvarchar](100) NULL,
	[source_record_id] [bigint] NULL,
	[title] [nvarchar](500) NOT NULL,
	[current_version_id] [uniqueidentifier] NULL,
	[status] [nvarchar](64) NOT NULL,
	[access_roles] [nvarchar](max) NULL,
	[checksum] [nvarchar](128) NULL,
	[metadata_json] [nvarchar](max) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_by] [nvarchar](128) NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_by] [nvarchar](128) NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_knowledge_document_source] UNIQUE NONCLUSTERED 
(
	[tenant_id] ASC,
	[module] ASC,
	[resource] ASC,
	[source_type] ASC,
	[source_record_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_knowledge_document_version]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_knowledge_document_version](
	[id] [uniqueidentifier] NOT NULL,
	[document_id] [uniqueidentifier] NOT NULL,
	[version_number] [int] NOT NULL,
	[checksum] [nvarchar](128) NULL,
	[storage_uri] [nvarchar](1024) NULL,
	[content_text] [nvarchar](max) NULL,
	[mime_type] [nvarchar](128) NULL,
	[language] [nvarchar](32) NULL,
	[parse_status] [nvarchar](64) NOT NULL,
	[parsed_at] [datetime2](7) NULL,
	[supersedes_version_id] [uniqueidentifier] NULL,
	[is_deleted] [bit] NOT NULL,
	[created_by] [nvarchar](128) NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_by] [nvarchar](128) NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_doc_version_document_version] UNIQUE NONCLUSTERED 
(
	[document_id] ASC,
	[version_number] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_mitre_technique]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_mitre_technique](
	[id] [uniqueidentifier] NOT NULL,
	[technique_id] [nvarchar](20) NOT NULL,
	[sub_technique_id] [nvarchar](20) NULL,
	[name] [nvarchar](255) NOT NULL,
	[tactic] [nvarchar](100) NULL,
	[description] [nvarchar](max) NULL,
	[url] [nvarchar](500) NULL,
	[is_deleted] [bit] NOT NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_mitre_technique_code] UNIQUE NONCLUSTERED 
(
	[technique_id] ASC,
	[sub_technique_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_model_cards]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_model_cards](
	[id] [uniqueidentifier] NOT NULL,
	[role] [nvarchar](128) NOT NULL,
	[provider] [nvarchar](256) NOT NULL,
	[model_name] [nvarchar](256) NOT NULL,
	[max_tokens] [int] NOT NULL,
	[cost_per_1k_input] [float] NOT NULL,
	[cost_per_1k_output] [float] NOT NULL,
	[latency_p50_ms] [float] NOT NULL,
	[latency_p99_ms] [float] NOT NULL,
	[description] [nvarchar](max) NULL,
	[is_deleted] [bit] NOT NULL,
	[updated_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_model_cards] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_model_cards_role] UNIQUE NONCLUSTERED 
(
	[role] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_ontology_concept]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_ontology_concept](
	[id] [uniqueidentifier] NOT NULL,
	[concept_id] [nvarchar](100) NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[name] [nvarchar](255) NOT NULL,
	[name_ar] [nvarchar](255) NULL,
	[keywords] [nvarchar](max) NULL,
	[keywords_ar] [nvarchar](max) NULL,
	[module_mappings] [nvarchar](max) NULL,
	[resource_mappings] [nvarchar](max) NULL,
	[parent_concept_id] [uniqueidentifier] NULL,
	[external_source] [nvarchar](100) NULL,
	[external_source_id] [nvarchar](255) NULL,
	[description] [nvarchar](max) NULL,
	[description_ar] [nvarchar](max) NULL,
	[version] [nvarchar](64) NOT NULL,
	[review_status] [nvarchar](64) NOT NULL,
	[access_roles] [nvarchar](max) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_by] [uniqueidentifier] NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_by] [uniqueidentifier] NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_ontology_concept_tenant_concept_id] UNIQUE NONCLUSTERED 
(
	[tenant_id] ASC,
	[concept_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_ontology_relationship]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_ontology_relationship](
	[id] [uniqueidentifier] NOT NULL,
	[source_concept_id] [uniqueidentifier] NOT NULL,
	[target_concept_id] [uniqueidentifier] NOT NULL,
	[relationship_type] [nvarchar](64) NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[weight] [decimal](10, 4) NULL,
	[external_source] [nvarchar](100) NULL,
	[version] [nvarchar](64) NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[created_by] [uniqueidentifier] NULL,
	[creation_date] [datetime2](7) NOT NULL,
	[updated_by] [uniqueidentifier] NULL,
	[updated_on] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_netra_ontology_relationship_tenant_pair] UNIQUE NONCLUSTERED 
(
	[tenant_id] ASC,
	[source_concept_id] ASC,
	[relationship_type] ASC,
	[target_concept_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_outbox_event]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_outbox_event](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[entity_id] [uniqueidentifier] NULL,
	[event_type] [nvarchar](128) NOT NULL,
	[aggregate_id] [uniqueidentifier] NULL,
	[correlation_id] [nvarchar](128) NULL,
	[module] [nvarchar](100) NULL,
	[resource] [nvarchar](100) NULL,
	[idempotency_key] [nvarchar](256) NULL,
	[schema_version] [nvarchar](64) NULL,
	[payload] [nvarchar](max) NOT NULL,
	[graph_sync_status] [nvarchar](64) NOT NULL,
	[vector_sync_status] [nvarchar](64) NOT NULL,
	[retry_count] [int] NOT NULL,
	[max_retries] [int] NOT NULL,
	[last_error] [nvarchar](max) NULL,
	[created_at] [datetime2](7) NOT NULL,
	[processed_at] [datetime2](7) NULL,
	[completed_at] [datetime2](7) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_policy_packs]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_policy_packs](
	[id] [uniqueidentifier] NOT NULL,
	[pack_id] [nvarchar](128) NOT NULL,
	[name] [nvarchar](256) NOT NULL,
	[description] [nvarchar](max) NOT NULL,
	[policies] [nvarchar](max) NOT NULL,
	[is_builtin] [bit] NOT NULL,
	[requires_approval] [bit] NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_policy_packs] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
 CONSTRAINT [UQ_policy_packs_pack_id] UNIQUE NONCLUSTERED 
(
	[pack_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_prompt_versions]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_prompt_versions](
	[id] [uniqueidentifier] NOT NULL,
	[prompt_id] [nvarchar](256) NOT NULL,
	[version] [int] NOT NULL,
	[content_hash] [nvarchar](128) NOT NULL,
	[content] [nvarchar](max) NOT NULL,
	[author] [nvarchar](256) NOT NULL,
	[approved_by] [nvarchar](256) NULL,
	[activated_at] [datetime2](7) NULL,
	[deactivated_at] [datetime2](7) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_prompt_versions] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_query_costs]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_query_costs](
	[query_id] [nvarchar](256) NOT NULL,
	[tenant_id] [nvarchar](256) NOT NULL,
	[total_cost_usd] [float] NOT NULL,
	[planning_cost_usd] [float] NOT NULL,
	[embedding_cost_usd] [float] NOT NULL,
	[generation_cost_usd] [float] NOT NULL,
	[reranking_cost_usd] [float] NOT NULL,
	[grounding_cost_usd] [float] NOT NULL,
	[input_tokens] [int] NOT NULL,
	[output_tokens] [int] NOT NULL,
	[model_role] [nvarchar](128) NOT NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_query_costs] PRIMARY KEY CLUSTERED 
(
	[query_id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_runtime_config]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_runtime_config](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [uniqueidentifier] NOT NULL,
	[config_key] [nvarchar](256) NOT NULL,
	[config_value] [nvarchar](max) NOT NULL,
	[description] [nvarchar](max) NULL,
	[updated_by] [uniqueidentifier] NULL,
	[updated_at] [datetime2](7) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[netra_chatbot_tenant_policy_activations]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[netra_chatbot_tenant_policy_activations](
	[id] [uniqueidentifier] NOT NULL,
	[tenant_id] [nvarchar](256) NOT NULL,
	[pack_id] [nvarchar](128) NOT NULL,
	[activated_at] [datetime2](7) NOT NULL,
	[deactivated_at] [datetime2](7) NULL,
	[is_deleted] [bit] NOT NULL,
	[created_at] [datetime2](7) NOT NULL,
 CONSTRAINT [PK_netra_chatbot_tenant_policy_activations] PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[netra_chatbot_audit_log] ADD  CONSTRAINT [DF_netra_audit_log_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_benchmark_results] ADD  DEFAULT (newid()) FOR [id]
GO
ALTER TABLE [dbo].[netra_chatbot_benchmark_results] ADD  DEFAULT (sysdatetimeoffset()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping] ADD  CONSTRAINT [DF_netra_chunk_concept_mapping_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping] ADD  CONSTRAINT [DF_netra_chunk_concept_mapping_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback] ADD  CONSTRAINT [DF_netra_conversation_feedback_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback] ADD  CONSTRAINT [DF_netra_conversation_feedback_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_message] ADD  CONSTRAINT [DF_netra_conversation_message_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_message] ADD  CONSTRAINT [DF_netra_conversation_message_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_session] ADD  CONSTRAINT [DF_netra_conversation_session_status]  DEFAULT ('active') FOR [status]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_session] ADD  CONSTRAINT [DF_netra_conversation_session_turn_count]  DEFAULT ((0)) FOR [turn_count]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_session] ADD  CONSTRAINT [DF_netra_conversation_session_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_session] ADD  CONSTRAINT [DF_netra_conversation_session_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_session] ADD  CONSTRAINT [DF_netra_conversation_session_updated_at]  DEFAULT (sysutcdatetime()) FOR [updated_at]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_policy_sensitive]  DEFAULT ((0)) FOR [policy_sensitive]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_status]  DEFAULT (N'PENDING') FOR [status]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_reviewer_response]  DEFAULT (N'') FOR [reviewer_response]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_responder]  DEFAULT (N'') FOR [responder]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_escalation_requests] ADD  CONSTRAINT [DF_escalation_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_experiments] ADD  CONSTRAINT [DF_experiments_traffic_split]  DEFAULT ((0.5)) FOR [traffic_split]
GO
ALTER TABLE [dbo].[netra_chatbot_experiments] ADD  CONSTRAINT [DF_experiments_status]  DEFAULT (N'RUNNING') FOR [status]
GO
ALTER TABLE [dbo].[netra_chatbot_experiments] ADD  CONSTRAINT [DF_experiments_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_experiments] ADD  CONSTRAINT [DF_experiments_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_idempotency_results] ADD  CONSTRAINT [DF_idempotency_response]  DEFAULT (N'{}') FOR [response_body]
GO
ALTER TABLE [dbo].[netra_chatbot_idempotency_results] ADD  CONSTRAINT [DF_idempotency_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_idempotency_results] ADD  CONSTRAINT [DF_idempotency_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_status]  DEFAULT ('running') FOR [status]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_records_processed]  DEFAULT ((0)) FOR [records_processed]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_records_failed]  DEFAULT ((0)) FOR [records_failed]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_chunks_created]  DEFAULT ((0)) FOR [chunks_created]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_embeddings_generated]  DEFAULT ((0)) FOR [embeddings_generated]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_graph_nodes_synced]  DEFAULT ((0)) FOR [graph_nodes_synced]
GO
ALTER TABLE [dbo].[netra_chatbot_ingestion_run] ADD  CONSTRAINT [DF_netra_ingestion_run_started_at]  DEFAULT (sysutcdatetime()) FOR [started_at]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk] ADD  CONSTRAINT [DF_netra_chunk_embedding_status]  DEFAULT ('pending') FOR [embedding_status]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk] ADD  CONSTRAINT [DF_netra_chunk_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk] ADD  CONSTRAINT [DF_netra_chunk_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document] ADD  CONSTRAINT [DF_netra_knowledge_document_status]  DEFAULT ('active') FOR [status]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document] ADD  CONSTRAINT [DF_netra_knowledge_document_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document] ADD  CONSTRAINT [DF_netra_knowledge_document_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version] ADD  CONSTRAINT [DF_netra_doc_version_parse_status]  DEFAULT ('pending') FOR [parse_status]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version] ADD  CONSTRAINT [DF_netra_doc_version_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version] ADD  CONSTRAINT [DF_netra_doc_version_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_mitre_technique] ADD  CONSTRAINT [DF_netra_mitre_technique_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_mitre_technique] ADD  CONSTRAINT [DF_netra_mitre_technique_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_model_cards] ADD  CONSTRAINT [DF_model_cards_description]  DEFAULT (N'') FOR [description]
GO
ALTER TABLE [dbo].[netra_chatbot_model_cards] ADD  CONSTRAINT [DF_model_cards_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_model_cards] ADD  CONSTRAINT [DF_model_cards_updated_at]  DEFAULT (sysutcdatetime()) FOR [updated_at]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept] ADD  CONSTRAINT [DF_netra_ontology_concept_version]  DEFAULT ('1.0') FOR [version]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept] ADD  CONSTRAINT [DF_netra_ontology_concept_review_status]  DEFAULT ('draft') FOR [review_status]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept] ADD  CONSTRAINT [DF_netra_ontology_concept_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept] ADD  CONSTRAINT [DF_netra_ontology_concept_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship] ADD  CONSTRAINT [DF_netra_ontology_relationship_version]  DEFAULT ('1.0') FOR [version]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship] ADD  CONSTRAINT [DF_netra_ontology_relationship_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship] ADD  CONSTRAINT [DF_netra_ontology_relationship_creation_date]  DEFAULT (sysutcdatetime()) FOR [creation_date]
GO
ALTER TABLE [dbo].[netra_chatbot_outbox_event] ADD  CONSTRAINT [DF_netra_outbox_graph_status]  DEFAULT ('PENDING') FOR [graph_sync_status]
GO
ALTER TABLE [dbo].[netra_chatbot_outbox_event] ADD  CONSTRAINT [DF_netra_outbox_vector_status]  DEFAULT ('PENDING') FOR [vector_sync_status]
GO
ALTER TABLE [dbo].[netra_chatbot_outbox_event] ADD  CONSTRAINT [DF_netra_outbox_retry_count]  DEFAULT ((0)) FOR [retry_count]
GO
ALTER TABLE [dbo].[netra_chatbot_outbox_event] ADD  CONSTRAINT [DF_netra_outbox_max_retries]  DEFAULT ((3)) FOR [max_retries]
GO
ALTER TABLE [dbo].[netra_chatbot_outbox_event] ADD  CONSTRAINT [DF_netra_outbox_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_description]  DEFAULT (N'') FOR [description]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_policies]  DEFAULT (N'[]') FOR [policies]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_is_builtin]  DEFAULT ((0)) FOR [is_builtin]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_requires_approval]  DEFAULT ((0)) FOR [requires_approval]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_policy_packs] ADD  CONSTRAINT [DF_policy_packs_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_prompt_versions] ADD  CONSTRAINT [DF_prompt_versions_approved_by]  DEFAULT (N'') FOR [approved_by]
GO
ALTER TABLE [dbo].[netra_chatbot_prompt_versions] ADD  CONSTRAINT [DF_prompt_versions_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_prompt_versions] ADD  CONSTRAINT [DF_prompt_versions_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_total_cost]  DEFAULT ((0.0)) FOR [total_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_planning_cost]  DEFAULT ((0.0)) FOR [planning_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_embedding_cost]  DEFAULT ((0.0)) FOR [embedding_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_generation_cost]  DEFAULT ((0.0)) FOR [generation_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_reranking_cost]  DEFAULT ((0.0)) FOR [reranking_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_grounding_cost]  DEFAULT ((0.0)) FOR [grounding_cost_usd]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_input_tokens]  DEFAULT ((0)) FOR [input_tokens]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_output_tokens]  DEFAULT ((0)) FOR [output_tokens]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_model_role]  DEFAULT (N'') FOR [model_role]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_query_costs] ADD  CONSTRAINT [DF_query_costs_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_runtime_config] ADD  CONSTRAINT [DF_netra_runtime_config_updated_at]  DEFAULT (sysutcdatetime()) FOR [updated_at]
GO
ALTER TABLE [dbo].[netra_chatbot_tenant_policy_activations] ADD  CONSTRAINT [DF_tenant_policy_act_activated]  DEFAULT (sysutcdatetime()) FOR [activated_at]
GO
ALTER TABLE [dbo].[netra_chatbot_tenant_policy_activations] ADD  CONSTRAINT [DF_tenant_policy_act_is_deleted]  DEFAULT ((0)) FOR [is_deleted]
GO
ALTER TABLE [dbo].[netra_chatbot_tenant_policy_activations] ADD  CONSTRAINT [DF_tenant_policy_act_created_at]  DEFAULT (sysutcdatetime()) FOR [created_at]
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping]  WITH CHECK ADD  CONSTRAINT [FK_netra_chunk_concept_mapping_chunk] FOREIGN KEY([chunk_id])
REFERENCES [dbo].[netra_chatbot_knowledge_chunk] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping] CHECK CONSTRAINT [FK_netra_chunk_concept_mapping_chunk]
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping]  WITH CHECK ADD  CONSTRAINT [FK_netra_chunk_concept_mapping_concept] FOREIGN KEY([concept_id])
REFERENCES [dbo].[netra_chatbot_ontology_concept] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_chunk_concept_mapping] CHECK CONSTRAINT [FK_netra_chunk_concept_mapping_concept]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback]  WITH CHECK ADD  CONSTRAINT [FK_netra_conversation_feedback_message] FOREIGN KEY([message_id])
REFERENCES [dbo].[netra_chatbot_conversation_message] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback] CHECK CONSTRAINT [FK_netra_conversation_feedback_message]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback]  WITH CHECK ADD  CONSTRAINT [FK_netra_conversation_feedback_session] FOREIGN KEY([conversation_id])
REFERENCES [dbo].[netra_chatbot_conversation_session] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback] CHECK CONSTRAINT [FK_netra_conversation_feedback_session]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_message]  WITH CHECK ADD  CONSTRAINT [FK_netra_conversation_message_session] FOREIGN KEY([session_id])
REFERENCES [dbo].[netra_chatbot_conversation_session] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_message] CHECK CONSTRAINT [FK_netra_conversation_message_session]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk]  WITH CHECK ADD  CONSTRAINT [FK_netra_chunk_document] FOREIGN KEY([document_id])
REFERENCES [dbo].[netra_chatbot_knowledge_document] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk] CHECK CONSTRAINT [FK_netra_chunk_document]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk]  WITH CHECK ADD  CONSTRAINT [FK_netra_chunk_document_version] FOREIGN KEY([document_version_id])
REFERENCES [dbo].[netra_chatbot_knowledge_document_version] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_chunk] CHECK CONSTRAINT [FK_netra_chunk_document_version]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document]  WITH CHECK ADD  CONSTRAINT [FK_netra_knowledge_document_current_version] FOREIGN KEY([current_version_id])
REFERENCES [dbo].[netra_chatbot_knowledge_document_version] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document] CHECK CONSTRAINT [FK_netra_knowledge_document_current_version]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version]  WITH CHECK ADD  CONSTRAINT [FK_netra_doc_version_document] FOREIGN KEY([document_id])
REFERENCES [dbo].[netra_chatbot_knowledge_document] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version] CHECK CONSTRAINT [FK_netra_doc_version_document]
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version]  WITH CHECK ADD  CONSTRAINT [FK_netra_doc_version_supersedes] FOREIGN KEY([supersedes_version_id])
REFERENCES [dbo].[netra_chatbot_knowledge_document_version] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_knowledge_document_version] CHECK CONSTRAINT [FK_netra_doc_version_supersedes]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept]  WITH CHECK ADD  CONSTRAINT [FK_netra_ontology_concept_parent] FOREIGN KEY([parent_concept_id])
REFERENCES [dbo].[netra_chatbot_ontology_concept] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_concept] CHECK CONSTRAINT [FK_netra_ontology_concept_parent]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship]  WITH CHECK ADD  CONSTRAINT [FK_netra_ontology_relationship_source] FOREIGN KEY([source_concept_id])
REFERENCES [dbo].[netra_chatbot_ontology_concept] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship] CHECK CONSTRAINT [FK_netra_ontology_relationship_source]
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship]  WITH CHECK ADD  CONSTRAINT [FK_netra_ontology_relationship_target] FOREIGN KEY([target_concept_id])
REFERENCES [dbo].[netra_chatbot_ontology_concept] ([id])
GO
ALTER TABLE [dbo].[netra_chatbot_ontology_relationship] CHECK CONSTRAINT [FK_netra_ontology_relationship_target]
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback]  WITH CHECK ADD  CONSTRAINT [CK_netra_conversation_feedback_rating] CHECK  (([rating]>=(1) AND [rating]<=(5)))
GO
ALTER TABLE [dbo].[netra_chatbot_conversation_feedback] CHECK CONSTRAINT [CK_netra_conversation_feedback_rating]
GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_admin_get_audit_logs]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_admin_get_audit_logs]
    @tenant_id UNIQUEIDENTIFIER,
    @limit INT = 50,
    @category NVARCHAR(64) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT TOP (@limit)
        created_at,
        user_id,
        session_id,
        action_category,
        action_type,
        entity_type,
        resource_id,
        correlation_id,
        metadata_json
    FROM netra_chatbot_audit_log WITH (NOLOCK)
    WHERE tenant_id = @tenant_id
      AND (@category IS NULL OR action_category = @category)
    ORDER BY created_at DESC;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_admin_get_stats]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_admin_get_stats]
    @tenant_id UNIQUEIDENTIFIER
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    DECLARE @tenant_id_str NVARCHAR(256) = CAST(@tenant_id AS NVARCHAR(256));

    -- Original 4 counters
    DECLARE @concepts INT, @chunks INT, @feedback INT, @outbox INT;

    SELECT @concepts = COUNT(*) FROM netra_chatbot_ontology_concept WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND is_deleted = 0;

    SELECT @chunks = COUNT(*) FROM netra_chatbot_knowledge_chunk WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND is_deleted = 0;

    SELECT @feedback = COUNT(*) FROM netra_chatbot_conversation_feedback WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND is_deleted = 0;

    SELECT @outbox = COUNT(*) FROM netra_chatbot_outbox_event WITH (NOLOCK)
    WHERE tenant_id = @tenant_id;

    -- Governance counters (8 new)
    DECLARE @query_costs INT = 0, @prompt_versions INT = 0, @policy_packs INT = 0;
    DECLARE @escalations_pending INT = 0, @experiments_running INT = 0;
    DECLARE @model_cards INT = 0, @graph_communities INT = 0, @community_summaries INT = 0;

    IF OBJECT_ID('dbo.netra_chatbot_query_costs', 'U') IS NOT NULL
        SELECT @query_costs = COUNT(*) FROM netra_chatbot_query_costs WITH (NOLOCK)
        WHERE tenant_id = @tenant_id_str AND is_deleted = 0;

    IF OBJECT_ID('dbo.netra_chatbot_prompt_versions', 'U') IS NOT NULL
        SELECT @prompt_versions = COUNT(*) FROM netra_chatbot_prompt_versions WITH (NOLOCK)
        WHERE is_deleted = 0;

    IF OBJECT_ID('dbo.netra_chatbot_policy_packs', 'U') IS NOT NULL
        SELECT @policy_packs = COUNT(*) FROM netra_chatbot_policy_packs WITH (NOLOCK)
        WHERE is_deleted = 0;

    IF OBJECT_ID('dbo.netra_chatbot_escalation_requests', 'U') IS NOT NULL
        SELECT @escalations_pending = COUNT(*) FROM netra_chatbot_escalation_requests WITH (NOLOCK)
        WHERE tenant_id = @tenant_id_str AND status = N'PENDING' AND is_deleted = 0;

    IF OBJECT_ID('dbo.netra_chatbot_experiments', 'U') IS NOT NULL
        SELECT @experiments_running = COUNT(*) FROM netra_chatbot_experiments WITH (NOLOCK)
        WHERE tenant_id = @tenant_id_str AND status = N'RUNNING' AND is_deleted = 0;

    IF OBJECT_ID('dbo.netra_chatbot_model_cards', 'U') IS NOT NULL
        SELECT @model_cards = COUNT(*) FROM netra_chatbot_model_cards WITH (NOLOCK)
        WHERE is_deleted = 0;

    IF OBJECT_ID('dbo.graph_communities', 'U') IS NOT NULL
        SELECT @graph_communities = COUNT(*) FROM graph_communities WITH (NOLOCK)
        WHERE tenant_id = @tenant_id_str AND is_deleted = 0;

    IF OBJECT_ID('dbo.community_summaries', 'U') IS NOT NULL
        SELECT @community_summaries = COUNT(*) FROM community_summaries WITH (NOLOCK)
        WHERE tenant_id = @tenant_id_str AND is_deleted = 0;

    SELECT
        @concepts            AS ontology_concepts,
        @chunks              AS knowledge_chunks,
        @feedback            AS feedback_entries,
        @outbox              AS outbox_messages,
        @query_costs         AS query_cost_records,
        @prompt_versions     AS prompt_versions,
        @policy_packs        AS policy_packs,
        @escalations_pending AS escalations_pending,
        @experiments_running AS experiments_running,
        @model_cards         AS model_cards,
        @graph_communities   AS graph_communities,
        @community_summaries AS community_summaries;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_admin_get_sync_status]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_admin_get_sync_status]
    @tenant_id UNIQUEIDENTIFIER
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @pending_graph INT, @pending_vector INT, @failed INT;
    DECLARE @oldest DATETIMEOFFSET;

    SELECT @pending_graph = COUNT(*) FROM netra_chatbot_outbox_event WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND graph_sync_status = 'PENDING';

    SELECT @pending_vector = COUNT(*) FROM netra_chatbot_outbox_event WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND vector_sync_status = 'PENDING';

    SELECT @failed = COUNT(*) FROM netra_chatbot_outbox_event WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND (graph_sync_status = 'FAILED' OR vector_sync_status = 'FAILED');

    SELECT TOP 1 @oldest = created_at FROM netra_chatbot_outbox_event WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND (graph_sync_status = 'PENDING' OR vector_sync_status = 'PENDING')
    ORDER BY created_at ASC;

    SELECT 
        @pending_graph AS pending_graph,
        @pending_vector AS pending_vector,
        @failed AS failed_count,
        @oldest AS oldest_pending_event;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_audit_log_insert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_audit_log_insert]
    @id UNIQUEIDENTIFIER,
    @action_category NVARCHAR(128),
    @action_type NVARCHAR(128),
    @tenant_id UNIQUEIDENTIFIER,
    @user_id UNIQUEIDENTIFIER,
    @scope_entity_id UNIQUEIDENTIFIER,
    @entity_type NVARCHAR(128),
    @resource_id NVARCHAR(128),
    @metadata_json NVARCHAR(MAX),
    @session_id NVARCHAR(128),
    @correlation_id NVARCHAR(128),
    @ip_address NVARCHAR(45),
    @user_agent NVARCHAR(512),
    @created_at DATETIME2(7),
    @before_state NVARCHAR(MAX) = NULL,
    @after_state NVARCHAR(MAX) = NULL,
    @previous_checksum NVARCHAR(128) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        INSERT INTO netra_chatbot_audit_log
            (id, action_category, action_type, tenant_id, user_id, scope_entity_id,
             entity_type, resource_id, metadata_json, session_id, correlation_id,
             ip_address, user_agent, created_at, before_state, after_state,
             previous_checksum)
        VALUES
            (@id, @action_category, @action_type, @tenant_id, @user_id, @scope_entity_id,
             @entity_type, @resource_id, @metadata_json, @session_id, @correlation_id,
             @ip_address, @user_agent, @created_at, @before_state, @after_state,
             @previous_checksum);
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_document_soft_delete]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_document_soft_delete]
    @tenant_id   UNIQUEIDENTIFIER,
    @document_id UNIQUEIDENTIFIER,
    @entity_id   UNIQUEIDENTIFIER = NULL   -- NULL = no entity filter (platform-admin)
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    BEGIN TRY
        BEGIN TRANSACTION;

            -- Guard: verify the document belongs to this tenant AND entity before
            -- returning any data or making any changes.  Without this check an
            -- entity could delete a document that belongs to a different entity
            -- within the same tenant.
            --
            -- entity_id matching rules:
            --   @entity_id IS NULL               â†’ no entity filter (platform-admin)
            --   document.entity_id IS NULL        â†’ shared/unscoped document; any entity may delete
            --   document.entity_id = @entity_id   â†’ owned by this entity; allowed
            --   document.entity_id != @entity_id  â†’ owned by a different entity; DENIED
            IF NOT EXISTS (
                SELECT 1
                FROM   netra_chatbot_knowledge_document WITH (NOLOCK)
                WHERE  id        = @document_id
                  AND  tenant_id = @tenant_id
                  AND  is_deleted = 0
                  AND  (@entity_id IS NULL OR entity_id IS NULL OR entity_id = @entity_id)
            )
            BEGIN
                -- Document not found or owned by a different entity.
                -- Roll back, return an empty chunk-id result set, and exit cleanly.
                ROLLBACK TRANSACTION;
                SELECT NULL AS chunk_id WHERE 1 = 0;
                RETURN;
            END

            -- 1. Return chunk IDs for vector-store cleanup in the application layer.
            --    Must come before the rows are marked deleted.
            --    Entity-scoped so cross-entity chunk IDs are never returned.
            SELECT chunk_id
            FROM netra_chatbot_knowledge_chunk WITH (NOLOCK)
            WHERE document_id = @document_id
              AND tenant_id   = @tenant_id
              AND is_deleted  = 0
              AND (@entity_id IS NULL OR entity_id IS NULL OR entity_id = @entity_id);

            -- 2. Soft-delete the document row (entity-scoped).
            UPDATE netra_chatbot_knowledge_document
            SET    is_deleted = 1,
                   updated_on = SYSUTCDATETIME()
            WHERE  id         = @document_id
              AND  tenant_id  = @tenant_id
              AND  is_deleted = 0
              AND  (@entity_id IS NULL OR entity_id = @entity_id);

            -- 3. Soft-delete ALL versions of the document.
            --    netra_chatbot_knowledge_document_version has no entity_id column;
            --    isolation is guaranteed by the ownership guard above.
            UPDATE netra_chatbot_knowledge_document_version
            SET    is_deleted = 1,
                   updated_on = SYSUTCDATETIME()
            WHERE  document_id = @document_id
              AND  is_deleted  = 0;

            -- 4. Soft-delete all chunks (entity-scoped for defense-in-depth).
            UPDATE netra_chatbot_knowledge_chunk
            SET    is_deleted = 1,
                   updated_on = SYSUTCDATETIME()
            WHERE  document_id = @document_id
              AND  tenant_id   = @tenant_id
              AND  is_deleted  = 0
              AND  (@entity_id IS NULL OR entity_id = @entity_id);

            -- 5. Cancel any pending or in-flight outbox events for this document
            --    so the graph/vector sync workers do not attempt to index a
            --    document that no longer exists.
            --    Entity-scoped so events belonging to other entities are untouched.
            --    PENDING  â†’ COMPLETED  (skip processing)
            --    CLAIMED  â†’ COMPLETED  (worker will see COMPLETED on re-check and no-op)
            UPDATE netra_chatbot_outbox_event
            SET    graph_sync_status  = CASE
                       WHEN graph_sync_status  IN ('PENDING', 'CLAIMED') THEN 'COMPLETED'
                       ELSE graph_sync_status
                   END,
                   vector_sync_status = CASE
                       WHEN vector_sync_status IN ('PENDING', 'CLAIMED') THEN 'COMPLETED'
                       ELSE vector_sync_status
                   END,
                   last_error    = 'Document deleted',
                   completed_at  = SYSUTCDATETIME()
            WHERE  tenant_id     = @tenant_id
              AND  aggregate_id  = @document_id
              AND  (@entity_id IS NULL OR entity_id = @entity_id)
              AND  (
                       graph_sync_status  IN ('PENDING', 'CLAIMED')
                    OR vector_sync_status IN ('PENDING', 'CLAIMED')
                   );

        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_feedback_insert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_feedback_insert]
    @id UNIQUEIDENTIFIER,
    @tenant_id UNIQUEIDENTIFIER,
    @entity_id UNIQUEIDENTIFIER,
    @user_id UNIQUEIDENTIFIER,
    @conversation_id UNIQUEIDENTIFIER,
    @message_id UNIQUEIDENTIFIER,
    @rating INT,
    @comment NVARCHAR(MAX),
    @is_deleted BIT,
    @created_at DATETIME2(7)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        INSERT INTO netra_chatbot_conversation_feedback
            (id, tenant_id, entity_id, user_id, conversation_id, message_id, rating, comment, is_deleted, created_at)
        VALUES 
            (@id, @tenant_id, @entity_id, @user_id, @conversation_id, @message_id, @rating, @comment, @is_deleted, @created_at);
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_message_get_history]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_message_get_history]
    @tenant_id UNIQUEIDENTIFIER,
    @session_id UNIQUEIDENTIFIER,
    @limit INT = 50
AS
BEGIN
    SET NOCOUNT ON;
    SELECT TOP (@limit) id, role, content, grounded, pipeline_metadata, created_at
    FROM netra_chatbot_conversation_message WITH (NOLOCK)
    WHERE tenant_id = @tenant_id AND session_id = @session_id AND is_deleted = 0
    ORDER BY created_at ASC;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_message_insert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_message_insert]
    @id UNIQUEIDENTIFIER,
    @tenant_id UNIQUEIDENTIFIER,
    @session_id UNIQUEIDENTIFIER,
    @role NVARCHAR(64),
    @content NVARCHAR(MAX),
    @grounded BIT,
    @pipeline_metadata NVARCHAR(MAX),
    @created_at DATETIME2(7)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        INSERT INTO netra_chatbot_conversation_message 
            (id, tenant_id, session_id, role, content, grounded, pipeline_metadata, created_at)
        VALUES 
            (@id, @tenant_id, @session_id, @role, @content, @grounded, @pipeline_metadata, @created_at);
        
        -- Touch the session timestamp
        UPDATE netra_chatbot_conversation_session SET updated_at = @created_at 
        WHERE id = @session_id AND tenant_id = @tenant_id;
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_outbox_claim]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_outbox_claim]
    @limit INT,
    @sync_type NVARCHAR(64)
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    DECLARE @ClaimedEvents TABLE (id UNIQUEIDENTIFIER);
    BEGIN TRY
        BEGIN TRANSACTION;
            IF @sync_type = 'graph'
            BEGIN
                INSERT INTO @ClaimedEvents (id)
                SELECT TOP (@limit) id FROM netra_chatbot_outbox_event WITH (UPDLOCK, READPAST)
                WHERE graph_sync_status IN ('PENDING', 'FAILED') AND (processed_at IS NULL OR processed_at <= SYSUTCDATETIME());
                UPDATE netra_chatbot_outbox_event SET graph_sync_status = 'CLAIMED' WHERE id IN (SELECT id FROM @ClaimedEvents);
            END
            ELSE IF @sync_type = 'vector'
            BEGIN
                INSERT INTO @ClaimedEvents (id)
                SELECT TOP (@limit) id FROM netra_chatbot_outbox_event WITH (UPDLOCK, READPAST)
                WHERE vector_sync_status IN ('PENDING', 'FAILED') AND (processed_at IS NULL OR processed_at <= SYSUTCDATETIME());
                UPDATE netra_chatbot_outbox_event SET vector_sync_status = 'CLAIMED' WHERE id IN (SELECT id FROM @ClaimedEvents);
            END
            ELSE
            BEGIN
                INSERT INTO @ClaimedEvents (id)
                SELECT TOP (@limit) id FROM netra_chatbot_outbox_event WITH (UPDLOCK, READPAST)
                WHERE (graph_sync_status IN ('PENDING', 'FAILED') OR vector_sync_status IN ('PENDING', 'FAILED'))
                  AND (processed_at IS NULL OR processed_at <= SYSUTCDATETIME());
                UPDATE netra_chatbot_outbox_event 
                SET graph_sync_status = CASE WHEN graph_sync_status IN ('PENDING', 'FAILED') THEN 'CLAIMED' ELSE graph_sync_status END,
                    vector_sync_status = CASE WHEN vector_sync_status IN ('PENDING', 'FAILED') THEN 'CLAIMED' ELSE vector_sync_status END
                WHERE id IN (SELECT id FROM @ClaimedEvents);
            END
            SELECT id, tenant_id, entity_id, event_type, aggregate_id, correlation_id, [module], [resource],
                idempotency_key, schema_version, payload, graph_sync_status, vector_sync_status, 
                retry_count, max_retries, last_error, created_at, processed_at, completed_at
            FROM netra_chatbot_outbox_event WITH (NOLOCK) WHERE id IN (SELECT id FROM @ClaimedEvents);
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_outbox_get_event]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_outbox_get_event]
    @id UNIQUEIDENTIFIER
AS
BEGIN
    SET NOCOUNT ON;
    SELECT id, tenant_id, entity_id, event_type, aggregate_id, correlation_id, [module], [resource],
        idempotency_key, schema_version, payload, graph_sync_status, vector_sync_status, 
        retry_count, max_retries, last_error, created_at, processed_at, completed_at
    FROM netra_chatbot_outbox_event WITH (NOLOCK) WHERE id = @id;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_outbox_insert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_outbox_insert]
    @id UNIQUEIDENTIFIER,
    @tenant_id UNIQUEIDENTIFIER,
    @entity_id UNIQUEIDENTIFIER,
    @event_type NVARCHAR(128),
    @aggregate_id UNIQUEIDENTIFIER,
    @correlation_id NVARCHAR(128),
    @module NVARCHAR(100),
    @resource NVARCHAR(100),
    @idempotency_key NVARCHAR(256),
    @schema_version NVARCHAR(64),
    @payload NVARCHAR(MAX),
    @graph_sync_status NVARCHAR(64) = 'PENDING',
    @vector_sync_status NVARCHAR(64) = 'PENDING',
    @retry_count INT = 0,
    @max_retries INT = 3,
    @last_error NVARCHAR(MAX) = NULL,
    @created_at DATETIME2(7) = NULL,
    @processed_at DATETIME2(7) = NULL,
    @completed_at DATETIME2(7) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        INSERT INTO netra_chatbot_outbox_event
            (id, tenant_id, entity_id, event_type, aggregate_id, correlation_id, [module], [resource], 
             idempotency_key, schema_version, payload, graph_sync_status, vector_sync_status, 
             retry_count, max_retries, last_error, created_at, processed_at, completed_at)
        VALUES 
            (@id, @tenant_id, @entity_id, @event_type, @aggregate_id, @correlation_id, @module, @resource, 
             @idempotency_key, @schema_version, @payload, @graph_sync_status, @vector_sync_status, 
             @retry_count, @max_retries, @last_error, @created_at, @processed_at, @completed_at);
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_resource_search]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_resource_search]
    @table_name NVARCHAR(128),
    @column_list NVARCHAR(MAX),
    @tenant_id UNIQUEIDENTIFIER,
    @entity_id UNIQUEIDENTIFIER,
    @include_shared BIT,
    @sort_field NVARCHAR(128),
    @sort_dir NVARCHAR(10),
    @offset INT,
    @limit INT,
    @where_clauses NVARCHAR(MAX)
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @ParamDefinition NVARCHAR(MAX);
    SET @ParamDefinition = N'@tenant_id UNIQUEIDENTIFIER, @entity_id UNIQUEIDENTIFIER, @offset INT, @limit INT';
    IF @column_list IS NULL OR @column_list = '' OR @column_list = '*' SET @column_list = 'id, tenant_id, entity_id';
    SET @sql = N'SELECT ' + @column_list + N' FROM ' + QUOTENAME(@table_name) + N' WITH (NOLOCK) WHERE tenant_id = @tenant_id AND is_deleted = 0';
    IF @entity_id IS NOT NULL BEGIN IF @include_shared = 1 SET @sql = @sql + N' AND (entity_id IS NULL OR entity_id = @entity_id)'; ELSE SET @sql = @sql + N' AND entity_id = @entity_id'; END
    IF @where_clauses IS NOT NULL AND @where_clauses <> '' SET @sql = @sql + N' AND ' + @where_clauses;
    SET @sql = @sql + N' ORDER BY ' + QUOTENAME(@sort_field) + N' ' + @sort_dir + N' OFFSET @offset ROWS FETCH NEXT @limit ROWS ONLY';
    DECLARE @count_sql NVARCHAR(MAX);
    SET @count_sql = N'SELECT COUNT(*) FROM ' + QUOTENAME(@table_name) + N' WITH (NOLOCK) WHERE tenant_id = @tenant_id AND is_deleted = 0';
    IF @entity_id IS NOT NULL BEGIN IF @include_shared = 1 SET @count_sql = @count_sql + N' AND (entity_id IS NULL OR entity_id = @entity_id)'; ELSE SET @count_sql = @count_sql + N' AND entity_id = @entity_id'; END
    IF @where_clauses IS NOT NULL AND @where_clauses <> '' SET @count_sql = @count_sql + N' AND ' + @where_clauses;
    BEGIN TRY
        EXEC sp_executesql @count_sql, @ParamDefinition, @tenant_id, @entity_id, @offset, @limit;
        EXEC sp_executesql @sql, @ParamDefinition, @tenant_id, @entity_id, @offset, @limit;
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_runtime_config_delete]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_runtime_config_delete]
    @tenant_id UNIQUEIDENTIFIER,
    @config_key NVARCHAR(256)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        DELETE FROM netra_chatbot_runtime_config
        WHERE tenant_id = @tenant_id AND config_key = @config_key;
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_runtime_config_get]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_runtime_config_get]
    @tenant_id UNIQUEIDENTIFIER,
    @config_key NVARCHAR(256)
AS
BEGIN
    SET NOCOUNT ON;
    SELECT config_value FROM netra_chatbot_runtime_config WITH (NOLOCK) WHERE tenant_id = @tenant_id AND config_key = @config_key;
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_runtime_config_upsert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_runtime_config_upsert]
    @id UNIQUEIDENTIFIER,
    @tenant_id UNIQUEIDENTIFIER,
    @config_key NVARCHAR(256),
    @config_value NVARCHAR(MAX),
    @description NVARCHAR(MAX),
    @updated_by UNIQUEIDENTIFIER,
    @updated_at DATETIME2(7)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        MERGE netra_chatbot_runtime_config AS target
        USING (SELECT @tenant_id AS tenant_id, @config_key AS config_key) AS source
        ON target.tenant_id = source.tenant_id AND target.config_key = source.config_key
        WHEN MATCHED THEN
            UPDATE SET config_value = @config_value, description = @description, updated_by = @updated_by, updated_at = @updated_at
        WHEN NOT MATCHED THEN
            INSERT (id, tenant_id, config_key, config_value, description, updated_by, updated_at)
            VALUES (@id, @tenant_id, @config_key, @config_value, @description, @updated_by, @updated_at);
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_session_delete]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_session_delete]
    @tenant_id UNIQUEIDENTIFIER,
    @session_id UNIQUEIDENTIFIER
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    BEGIN TRY
        BEGIN TRANSACTION;
            -- We follow the soft-delete pattern if columns exist, otherwise hard delete
            UPDATE netra_chatbot_conversation_message SET is_deleted = 1 
            WHERE tenant_id = @tenant_id AND session_id = @session_id;

            UPDATE netra_chatbot_conversation_session SET is_deleted = 1 
            WHERE id = @session_id AND tenant_id = @tenant_id;
            
            SELECT @@ROWCOUNT AS DeletedCount;
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
/****** Object:  StoredProcedure [dbo].[sp_netra_chatbot_session_upsert]    Script Date: 14-04-2026 15:04:11 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER OFF
GO

CREATE PROCEDURE [dbo].[sp_netra_chatbot_session_upsert]
    @id UNIQUEIDENTIFIER,
    @tenant_id UNIQUEIDENTIFIER,
    @entity_id UNIQUEIDENTIFIER,
    @user_id UNIQUEIDENTIFIER,
    @title NVARCHAR(512) = NULL,
    @updated_at DATETIME2(7)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        MERGE netra_chatbot_conversation_session WITH (HOLDLOCK) AS target
        USING (SELECT @id AS id, @tenant_id AS tenant_id) AS source
        ON target.id = source.id AND target.tenant_id = source.tenant_id
        WHEN MATCHED THEN
            UPDATE SET updated_at = @updated_at, title = COALESCE(@title, target.title)
        WHEN NOT MATCHED THEN
            INSERT (id, tenant_id, entity_id, user_id, title, created_at, updated_at)
            VALUES (@id, @tenant_id, @entity_id, @user_id, @title, @updated_at, @updated_at);
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
        RAISERROR(@ErrorMessage, 16, 1);
    END CATCH
END;

GO
