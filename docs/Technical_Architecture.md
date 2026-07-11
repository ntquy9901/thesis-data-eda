# Technical Architecture Document
# Vietnam Stock Market News-Price Correlation Analysis System

**Document Version:** 1.2
**Last Updated:** 2026-07-11
**Project:** Vietnam Stock Market Analysis
**Architecture Style:** Data Lakehouse with Pipeline Processing

---

## 1. System Overview

### 1.1 Architecture Philosophy
The system implements a **Data Lakehouse Architecture** combining the best features of data lakes (schema-on-read, flexibility) and data warehouses (schema-on-write, reliability). This architecture supports the Bronze/Silver/Gold data quality model while maintaining performance and reliability.

### 1.2 Core Principles
- **Data Quality First:** Three-layer quality model (Bronze/Silver/Gold)
- **Incremental Processing:** Only process new/changed data
- **Schema Enforcement:** Strict schema validation at each layer
- **Performance Optimization:** Columnar storage, partitioning, caching
- **Quality Compliance:** BMad framework enforces CLAUDE.md standards
- **Reproducibility:** Complete data lineage and version tracking

### 1.3 System Boundaries
```
┌─────────────────────────────────────────────────────────────┐
│                    External Data Sources                    │
│  ┌──────────────────┐      ┌──────────────────────────┐   │
│  │ News Crawl Data  │      │  Price Crawl Data         │   │
│  │  (User Provided) │      │  (User Provided)          │   │
│  └──────────────────┘      └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Lakehouse Pipeline                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Bronze Layer (Raw Data Preservation)               │  │
│  │  - Immutable source data                             │  │
│  │  - Original format (CSV)                             │  │
│  │  - Import logging & validation                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Silver Layer (Cleaned & Validated)                  │  │
│  │  - Data quality processing                            │  │
│  │  - Schema standardization                            │  │
│  │  - Parquet format with compression                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Gold Layer (Feature-Rich Analysis Ready)            │  │
│  │  - Feature engineering                                │  │
│  │  - Performance optimized                             │  │
│  │  - Analysis-ready datasets                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Analysis & Reporting                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Correlation  │  │ Statistical  │  │ Visualization │    │
│  │  Analysis    │  │   Testing    │  │   & Reports   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Data Pipeline Components

#### 2.1.1 Bronze Importer
**Purpose:** Import raw data from sources to Bronze layer
**Technology:** Python, Pandas, file system operations
**Input:** Raw data from crawl directories
**Output:** Immutable files in Bronze layer
**Key Features:**
- Schema validation on import
- File integrity verification
- Import logging and metadata tracking
- Duplicate detection (based on timestamps)

#### 2.1.2 Silver Processor
**Purpose:** Clean and validate Bronze data for Silver layer
**Technology:** Python, Pandas, NLP libraries
**Input:** Bronze layer data
**Output:** Cleaned Parquet files in Silver layer
**Key Features:**
- Vietnamese text processing (underthesea)
- Sentiment analysis (transformers)
- Technical indicator calculation
- Data quality scoring
- Schema enforcement

#### 2.1.3 Gold Feature Engineer
**Purpose:** Create analysis-ready features from Silver data
**Technology:** Python, Polars (for large datasets), Scikit-learn
**Input:** Silver layer data
**Output:** Feature-rich Parquet files in Gold layer
**Key Features:**
- Text embeddings (sentence-transformers)
- Advanced time-series features
- Correlation features
- Performance optimization
- Query optimization

#### 2.1.4 Analysis Engine
**Purpose:** Execute correlation and statistical analysis
**Technology:** Python, Statsmodels, ARCH, Scipy
**Input:** Gold layer features
**Output:** Analysis results and reports
**Key Features:**
- Correlation analysis (Pearson, Spearman)
- Lead-lag analysis (cross-correlation)
- Granger causality testing
- Event study methodology
- Statistical significance testing

### 2.2 Supporting Components

#### 2.2.1 Quality Validator
**Purpose:** Ensure data quality at each layer
**Technology:** Python, Pandas, custom validation rules
**Key Functions:**
- Schema validation
- Completeness checks
- Consistency verification
- Quality score calculation
- Alert generation

#### 2.2.2 Metadata Manager
**Purpose:** Track data lineage and versioning
**Technology:** Python, JSON, YAML
**Key Functions:**
- Data lineage tracking
- Version history management
- Schema documentation
- Processing logs
- Quality metrics storage

#### 2.2.3 Refresh Manager
**Purpose:** Handle daily data updates
**Technology:** Python, file system monitoring
**Key Functions:**
- New data detection
- Incremental processing
- Version tracking
- Rollback capability
- Status dashboard

---

## 3. Data Architecture

### 3.1 Lakehouse Layer Specifications

#### 3.1.1 Bronze Layer (Raw Data)
**Characteristics:**
- **Format:** Original (CSV for compatibility)
- **Compression:** None (preserve original format)
- **Schema:** Source schemas preserved
- **Update Strategy:** Append-only (immutable)
- **Retention:** Indefinite
- **Partitioning:** By date (YYYY/MM/DD)

**Directory Structure:**
```
bronze/
├── news/
│   ├── news_articles.csv          # Consolidated news
│   ├── cafef_articles.csv         # Individual sources
│   ├── ssi_articles.csv
│   ├── vndirect_articles.csv
│   ├── vnstock_articles.csv
│   └── hsc_articles.csv
├── prices/
│   ├── VCB_ohlcv.csv              # Individual stock prices
│   ├── VIC_ohlcv.csv
│   └── ... (30 stocks total)
└── _raw_imports/                  # Import logs and metadata
```

**Quality Rules:**
- File integrity verification (checksums)
- Source format validation
- Import timestamp recording
- No modification of source data

#### 3.1.2 Silver Layer (Cleaned Data)
**Characteristics:**
- **Format:** Parquet (columnar storage)
- **Compression:** Snappy (balanced speed/compression)
- **Schema:** Standardized schemas enforced
- **Update Strategy:** Incremental with full rebuild capability
- **Retention:** Current + previous version
- **Partitioning:** By source and date (source/YYYY/MM/DD)

**Directory Structure:**
```
silver/
├── news/
│   └── news_cleaned.parquet       # Single consolidated file
├── prices/
│   └── prices_cleaned.parquet    # Single consolidated file
└── _validated/                   # Quality reports and metrics
    ├── news_quality_report.json
    └── prices_quality_report.json
```

**Quality Rules:**
- Completeness: <20% missing data for critical fields
- Consistency: No duplicates, standardized formats
- Accuracy: Price relationships validated (high ≥ low)
- Quality Score: >80% for promotion to Gold

#### 3.1.3 Gold Layer (Feature-Rich Data)
**Characteristics:**
- **Format:** Parquet with statistics
- **Compression:** Zstandard (high compression ratio)
- **Schema:** Analysis-optimized schemas
- **Update Strategy:** Incremental feature updates
- **Retention:** Current version with feature history
- **Partitioning:** By ticker and period (ticker/YYYY/MM)

**Directory Structure:**
```
gold/
├── news/
│   ├── news_features.parquet      # News features with embeddings
│   └── sentiment_features.parquet # Sentiment analysis results
├── prices/
│   ├── price_features.parquet     # Price features and indicators
│   └── volatility_features.parquet # Volatility metrics
├── correlation/
│   ├── news_price_features.parquet  # Correlation features
│   └── news_volatility_features.parquet # Volatility features
└── analysis/
    ├── correlation_results.parquet # Pre-computed correlations
    └── statistical_tests.parquet   # Test results
```

**Quality Rules:**
- Feature completeness: 100% for required features
- Performance: Query time <1 second for typical queries
- Validated: All features statistically validated
- Documentation: Complete feature descriptions

### 3.2 Schema Definitions

#### 3.2.1 News Schema Evolution
```
Bronze: source, title, category, pub_date, url, author, lead, pdf_url, collected_at
         ↓ (add processing fields)
Silver: +article_id, content, sentiment_score, tickers_mentioned, word_count, 
         processed_at, quality_score
         ↓ (add feature fields)
Gold:   +sentiment_category, text_embedding[], topic_distribution[], 
         dominant_topic, market_impact_score, urgency_score
```

#### 3.2.2 Price Schema Evolution
```
Bronze: date, open, high, low, close, volume
         ↓ (add processing fields)
Silver: +ticker, returns, log_returns, volatility_20d, volume_ma_20d,
         processed_at, quality_score
         ↓ (add feature fields)
Gold:   +realized_volatility_5d, realized_volatility_20d, volatility_percentile_60d,
         volume_ratio, price_momentum_5d, price_momentum_20d, rsi_14, macd,
         bollinger_position
```

---

## 4. Processing Architecture

### 4.1 Pipeline Stages

#### Stage 1: Data Import (Bronze)
```python
# Pseudo-code
def bronze_import_pipeline():
    # Check for new data
    if new_data_available():
        # Import from sources
        news_df = load_from_source(news_source)
        prices_df = load_from_source(price_source)
        
        # Validate integrity
        validate_file_integrity(news_df, prices_df)
        
        # Save to Bronze (immutable)
        save_to_bronze(news_df, prices_df)
        
        # Log import
        log_import_operation(news_df, prices_df)
```

#### Stage 2: Data Processing (Silver)
```python
# Pseudo-code
def silver_processing_pipeline():
    # Load from Bronze
    news_bronze = load_from_bronze('news')
    prices_bronze = load_from_bronze('prices')
    
    # Process news
    news_silver = process_news(news_bronze)
    news_silver['sentiment'] = analyze_sentiment(news_silver['content'])
    news_silver['tickers'] = extract_tickers(news_silver['content'])
    
    # Process prices
    prices_silver = process_prices(prices_bronze)
    prices_silver['returns'] = calculate_returns(prices_silver)
    prices_silver['volatility'] = calculate_volatility(prices_silver)
    
    # Validate quality
    validate_data_quality(news_silver, prices_silver)
    
    # Save to Silver
    save_to_silver(news_silver, prices_silver)
```

#### Stage 3: Feature Engineering (Gold)
```python
# Pseudo-code
def gold_feature_pipeline():
    # Load from Silver
    news_silver = load_from_silver('news')
    prices_silver = load_from_silver('prices')
    
    # Engineer news features
    news_gold = engineer_news_features(news_silver)
    news_gold['embeddings'] = generate_embeddings(news_gold['content'])
    news_gold['topics'] = extract_topics(news_gold['content'])
    
    # Engineer price features
    prices_gold = engineer_price_features(prices_silver)
    prices_gold['momentum'] = calculate_momentum(prices_gold)
    prices_gold['technical'] = calculate_technical_indicators(prices_gold)
    
    # Engineer correlation features
    correlation_gold = engineer_correlation_features(news_gold, prices_gold)
    
    # Save to Gold
    save_to_gold(news_gold, prices_gold, correlation_gold)
```

### 4.2 Incremental Processing Strategy

#### 4.2.1 New Data Detection
```python
def detect_new_data():
    # Check file modification timestamps
    news_mod_time = get_file_modtime(news_source)
    price_mod_time = get_file_modtime(price_source)
    
    # Compare with last import time
    last_import = get_last_import_timestamp()
    
    # Determine if refresh needed
    news_need_refresh = news_mod_time > last_import
    prices_need_refresh = price_mod_time > last_import
    
    return news_need_refresh, prices_need_refresh
```

#### 4.2.2 Incremental Feature Update
```python
def incremental_feature_update():
    # Get date range of new data
    new_dates = get_new_data_dates()
    
    # Load existing features
    existing_features = load_gold_features()
    
    # Process only new dates
    new_features = process_new_dates(new_dates)
    
    # Merge with existing
    updated_features = merge_features(existing_features, new_features)
    
    # Save updated features
    save_gold_features(updated_features)
```

### 4.3 Data Quality Pipeline

#### 4.3.1 Quality Checks
```python
def quality_check_pipeline(data, layer):
    # Completeness check
    completeness = check_completeness(data)
    
    # Consistency check
    consistency = check_consistency(data)
    
    # Accuracy check
    accuracy = check_accuracy(data)
    
    # Uniqueness check
    uniqueness = check_uniqueness(data)
    
    # Calculate overall quality score
    quality_score = calculate_quality_score(
        completeness, consistency, accuracy, uniqueness
    )
    
    # Generate quality report
    report = generate_quality_report(quality_score, layer)
    
    # Determine if data can be promoted
    can_promote = quality_score > PROMOTION_THRESHOLD[layer]
    
    return can_promote, report
```

---

## 5. Technology Stack

### 5.1 Core Technologies

#### 5.1.1 Data Processing
- **Pandas:** Data manipulation and analysis
- **Polars:** High-performance data processing (large datasets)
- **NumPy:** Numerical computing
- **Apache Arrow:** Memory format and interoperability

#### 5.1.2 NLP & Text Processing
- **Underthesea:** Vietnamese tokenization and NLP
- **Transformers:** Multilingual models (PhoBERT)
- **Sentence-Transformers:** Text embeddings (multilingual)
- **Scikit-learn:** Machine learning utilities

#### 5.1.3 Financial Analysis
- **ARCH:** GARCH volatility models
- **Statsmodels:** Statistical testing and time series
- **Scipy:** Scientific computing
- **TA-Lib:** Technical analysis indicators (optional)

#### 5.1.4 Storage & I/O
- **Parquet:** Columnar storage format
- **PyArrow:** Parquet I/O optimizations
- **JSON/YAML:** Configuration and metadata
- **CSV:** Bronze layer compatibility

#### 5.1.5 Quality & Compliance
- **BMad:** Project management and quality enforcement
- **CLAUDE.md:** Quality standards and behavioral guidelines
- **Pytest:** Testing framework
- **Ruff/Mypy:** Linting and type checking

### 5.2 System Requirements

#### 5.2.1 Hardware Requirements
- **CPU:** 4+ cores recommended
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 50GB minimum (SSD recommended)
- **Network:** Local processing (no internet required after setup)

#### 5.2.2 Software Requirements
- **Operating System:** Windows 10/11, Linux, macOS
- **Python:** 3.10 or higher
- **Package Manager:** uv (recommended) or pip
- **Git:** For version control

---

## 6. Integration Architecture

### 6.1 External System Integration

#### 6.1.1 Data Source Integration
```
User Crawl Data → Bronze Import → Lakehouse
     ↓                    ↓
  (Provided)        (Immutable Copy)
```

**Integration Points:**
- **News Data:** `D:/bmad-projects/crawl_data/data`
- **Price Data:** `D:/bmad-projects/stock_vol_prediction01/data/raw`
- **Access Method:** File system read-only
- **Update Frequency:** Daily (user-driven)
- **Error Handling:** Graceful degradation if sources unavailable

#### 6.1.2 BMad Integration
```
BMad Framework → Quality Enforcement → Pipeline
      ↓                  ↓               ↓
  Config/Rules    CLAUDE.md checks   Compliant processing
```

**Integration Points:**
- **Configuration:** `.bmad/config.yaml`
- **Module Management:** `.bmad/module.yaml`
- **Quality Checks:** Automated compliance verification
- **Documentation:** Auto-generated from configs

### 6.2 Internal Component Integration

#### 6.2.1 Pipeline Integration
```
Refresh Manager → Bronze Import → Silver Process → Gold Features → Analysis
       ↓              ↓              ↓              ↓            ↓
  New Data      Raw Cleaned    Validated      Feature-rich   Insights
  Detection      Storage        Storage        Storage       Storage
```

#### 6.2.2 Data Flow Integration
```python
# Integrated pipeline execution
def execute_full_pipeline():
    # 1. Check for new data
    if needs_refresh():
        # 2. Import to Bronze
        bronze_importer.import_all()
        
        # 3. Process to Silver
        silver_processor.process_all()
        
        # 4. Engineer Gold features
        gold_features.engineer_all()
        
        # 5. Run analysis
        analysis_engine.run_analysis()
        
        # 6. Generate reports
        reporting.generate_reports()
```

---

## 7. Performance Architecture

### 7.1 Performance Optimization Strategies

#### 7.1.1 Storage Optimization
- **Columnar Storage:** Parquet format for efficient column access
- **Compression:** Snappy (Silver) and Zstandard (Gold) for space efficiency
- **Partitioning:** Date-based partitioning for efficient filtering
- **Statistics:** Parquet file statistics for query optimization

#### 7.1.2 Processing Optimization
- **Incremental Processing:** Only process new/changed data
- **Parallel Processing:** Multi-core processing for independent operations
- **Memory Management:** Chunked processing for large datasets
- **Caching:** Cache frequently accessed features

#### 7.1.3 Query Optimization
- **Predicate Pushdown:** Filter at storage level when possible
- **Column Pruning:** Only read required columns
- **Partition Pruning:** Skip irrelevant partitions
- **Materialized Views:** Pre-compute aggregations

### 7.2 Performance Targets

| Operation | Target | Maximum |
|----------|--------|---------|
| Bronze Import (1 day) | <5 min | <10 min |
| Silver Processing (1 day) | <10 min | <20 min |
| Gold Features (1 day) | <15 min | <30 min |
| Correlation Analysis (30 stocks) | <2 min | <5 min |
| Full Pipeline (daily) | <30 min | <60 min |
| Gold Layer Query | <1 sec | <5 sec |

---

## 8. Security Architecture

### 8.1 Data Security

#### 8.1.1 Access Control
- **Read-Only Sources:** Source directories are read-only
- **Local Processing:** No data transmission over network
- **File Permissions:** Restrictive permissions for sensitive data
- **Audit Logging:** All data access logged

#### 8.1.2 Data Integrity
- **Checksum Verification:** Verify file integrity on import
- **Immutable Bronze:** Bronze layer data never modified
- **Version Control:** Track all data versions
- **Validation:** Multi-stage validation pipeline

### 8.2 Code Security

#### 8.2.1 Quality Enforcement
- **CLAUDE.md Compliance:** All code follows behavioral guidelines
- **Code Review:** Required for all changes
- **Testing:** >80% coverage requirement
- **Linting:** Automated linting and type checking

#### 8.2.2 Dependency Management
- **Pinned Versions:** All dependencies pinned in requirements.txt
- **Security Scanning:** Regular vulnerability scans
- **Updates:** Controlled update process
- **Alternatives:** Fallback options for critical dependencies

---

## 9. Monitoring & Observability

### 9.1 Monitoring Components

#### 9.1.1 Pipeline Monitoring
- **Import Status:** Daily import success/failure
- **Processing Metrics:** Time taken for each stage
- **Data Quality:** Quality scores and trends
- **Storage Usage:** Disk space and file counts

#### 9.1.2 Data Quality Monitoring
- **Completeness:** Missing data ratios
- **Consistency:** Schema violations
- **Timeliness:** Data age and freshness
- **Accuracy:** Quality score trends

#### 9.1.3 Performance Monitoring
- **Processing Time:** Stage-by-stage timing
- **Resource Usage:** CPU, memory, disk I/O
- **Query Performance:** Gold layer query times
- **System Health:** Overall system status

### 9.2 Logging Strategy

#### 9.2.1 Pipeline Logs
```
data_lakehouse/_logs/
├── pipeline_20260711.log        # Daily pipeline logs
├── import_20260711.log          # Import specific logs
├── quality_20260711.log         # Quality check logs
└── performance_20260711.log      # Performance metrics
```

#### 9.2.2 Log Contents
- **Timestamp:** ISO 8601 format
- **Stage:** Pipeline stage name
- **Status:** Success/failure/partial
- **Metrics:** Key performance indicators
- **Errors:** Error messages and stack traces

### 9.3 Alerting Strategy

#### 9.3.1 Alert Conditions
- **Import Failure:** Daily import fails
- **Quality Drop:** Quality score drops below threshold
- **Performance Degradation:** Processing exceeds maximum time
- **Storage Issues:** Disk space >90% full

#### 9.3.2 Alert Channels
- **Log Files:** Written to log files
- **Status Dashboard:** Pipeline status dashboard
- **Quality Reports:** Generated quality reports
- **User Notifications:** Future enhancement

---

## 10. Deployment Architecture

### 10.1 Deployment Strategy

#### 10.1.1 Development Environment
- **Location:** Local development machine
- **Data:** Sample datasets for testing
- **Configuration:** Development settings
- **Purpose:** Feature development and testing

#### 10.1.2 Production Environment
- **Location:** Local machine or server
- **Data:** Full production datasets
- **Configuration:** Production settings
- **Purpose:** Daily analysis operation

### 10.2 Installation Process

#### 10.2.1 Initial Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd data_eda

# 2. Install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 3. Verify installation
pytest -m smoke
python .bmad/manager.py

# 4. Initial data import
python -m src.data.lakehouse_pipeline
```

#### 10.2.2 Daily Operation
```bash
# 1. Check data status
python -c "from src.data.refresh_data import get_data_status; print(get_data_status())"

# 2. Run pipeline if needed
python -c "from src.data.lakehouse_pipeline import LakehousePipeline; pipeline = LakehousePipeline(); pipeline.run_daily()"

# 3. Verify results
python -c "from src.data.lakehouse_pipeline import check_pipeline_status; print(check_pipeline_status())"
```

---

## 11. Maintenance Architecture

### 11.1 Maintenance Procedures

#### 11.1.1 Daily Maintenance
- **Data Status Check:** Verify data freshness
- **Pipeline Execution:** Run daily pipeline
- **Quality Verification:** Check data quality scores
- **Report Generation:** Generate daily reports

#### 11.1.2 Weekly Maintenance
- **Performance Review:** Analyze pipeline performance
- **Storage Management:** Clean up temporary files
- **Quality Trends:** Review quality score trends
- **Documentation Update:** Update documentation as needed

#### 11.1.3 Monthly Maintenance
- **System Review:** Overall system health check
- **Dependency Update:** Review and update dependencies
- **Archive Management:** Archive old logs and reports
- **Backup Verification:** Verify backup integrity

### 11.2 Troubleshooting Guide

#### 11.2.1 Common Issues

**Issue: Pipeline execution fails**
```
Diagnosis:
1. Check data sources are accessible
2. Verify file permissions
3. Check disk space
4. Review error logs

Resolution:
1. Ensure data sources are mounted/accessible
2. Fix file permissions if needed
3. Clean up disk space
4. Follow error log guidance
```

**Issue: Data quality score drops**
```
Diagnosis:
1. Check source data quality
2. Verify processing pipeline
3. Review quality rules
4. Analyze quality report

Resolution:
1. Fix source data issues
2. Update processing logic
3. Adjust quality rules if appropriate
4. Re-process affected data
```

**Issue: Performance degradation**
```
Diagnosis:
1. Check data volume growth
2. Analyze query patterns
3. Review system resources
4. Check for data skew

Resolution:
1. Optimize partitioning
2. Rebuild statistics
3. Add caching
4. Clean up old data
```

---

## 12. Future Architecture Considerations

### 12.1 Scalability Enhancements

#### 12.1.1 Horizontal Scaling
- **Distributed Processing:** Future support for multi-node processing
- **Data Sharding:** Shard data by ticker or date range
- **Load Balancing:** Distribute processing across nodes
- **Fault Tolerance:** Redundant processing capabilities

#### 12.1.2 Cloud Migration
- **Cloud Storage:** Migrate to cloud-based lakehouse (Delta Lake)
- **Serverless Processing:** AWS Lambda or similar for pipeline stages
- **Managed Services:** Use managed services for reduced maintenance
- **Hybrid Approach:** On-premises development, cloud production

### 12.2 Feature Enhancements

#### 12.2.1 Advanced Analytics
- **Machine Learning:** Add predictive modeling capabilities
- **Real-time Processing:** Stream processing for real-time analysis
- **Alternative Data:** Incorporate alternative data sources
- **Advanced NLP:** More sophisticated text analysis

#### 12.2.2 User Interface
- **Web Dashboard:** Interactive dashboard for monitoring
- **API Access:** RESTful API for data access
- **Alerting System:** Proactive alerting for anomalies
- **Collaboration:** Multi-user support and collaboration features

---

## 13. Architecture Decision Records

### 13.1 Key Decisions

#### ADR-001: Data Lakehouse Architecture
**Date:** 2026-07-11
**Status:** Accepted
**Decision:** Implement Bronze/Silver/Gold data lakehouse architecture
**Rationale:**
- Balances data lake flexibility with warehouse reliability
- Supports data quality requirements
- Enables incremental processing
- Provides clear data lineage
**Consequences:**
- + Strong data quality enforcement
- + Clear separation of concerns
- + Reproducible processing
- - More complex than single-layer approach
- - Requires additional storage

#### ADR-002: Parquet Format for Silver/Gold
**Date:** 2026-07-11
**Status:** Accepted
**Decision:** Use Parquet format for Silver and Gold layers
**Rationale:**
- Columnar storage for efficient queries
- Built-in compression
- Statistics for query optimization
- Wide language support
**Consequences:**
- + Better query performance
- + Reduced storage requirements
- + Schema enforcement
- - Additional dependency
- - Less human-readable than CSV

#### ADR-003: Local-First Architecture
**Date:** 2026-07-11
**Status:** Accepted
**Decision:** Design for local deployment without cloud dependencies
**Rationale:**
- User provides local data sources
- Reduces complexity
- No network latency
- Data security
**Consequences:**
- + Simpler deployment
- + Better performance
- + Data privacy
- - Limited scalability
- - Manual backup required

---

## 14. Appendix

### 14.1 Technical Diagrams

#### 14.1.1 Data Flow Diagram
```
┌────────────┐
│  External  │
│   Sources  │
└─────┬──────┘
      │
      ▼
┌───────────────────────────────────────┐
│         Bronze Import                  │
│  • Schema Validation                   │
│  • Integrity Check                     │
│  • Import Logging                      │
└─────┬───────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────┐
│      Silver Processing                │
│  • NLP Processing                      │
│  • Technical Indicators                │
│  • Quality Validation                  │
└─────┬───────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────┐
│     Gold Feature Engineering          │
│  • Embeddings                         │
│  • Time-Series Features               │
│  • Correlation Features                │
└─────┬───────────────────────────────────┘
      │
      ▼
┌───────────────────────────────────────┐
│       Analysis & Reporting             │
│  • Correlation Analysis                │
│  • Statistical Testing                 │
│  • Visualization                       │
└───────────────────────────────────────┘
```

### 14.2 Configuration Examples

#### 14.2.1 Pipeline Configuration
```yaml
pipeline:
  stages:
    - name: bronze_import
      enabled: true
      timeout: 600  # 10 minutes
    - name: silver_processing
      enabled: true
      timeout: 1200  # 20 minutes
    - name: gold_features
      enabled: true
      timeout: 1800  # 30 minutes

quality:
  thresholds:
    bronze: 0.0    # No threshold for raw data
    silver: 0.8    # 80% quality required
    gold: 0.9      # 90% quality required

performance:
  max_processing_time: 3600  # 1 hour
  max_memory_usage: "16GB"
  parallel_workers: 4
```

---

## 15. EDA Findings Integration (2025-2026 Analysis)

**Analysis Period:** 2025-01-02 to 2026-08-06
**Integration Date:** 2026-07-11
**Data Source:** Vietnam Stock Market News (SSI, CafeF, VNDirect)

### 15.1 Architecture Validation Based on EDA

#### Processing Architecture Validation

**Real-time Processing Confirmed:**
- **Current Volume:** 1.4 articles/day
- **Processing Time:** <5 minutes actual (vs 30 min target)
- **Validation:** ✅ Real-time architecture appropriate for current volume
- **Scalability:** Supports 3x growth projection (4.2 articles/day)

**Midnight Batch Processing Optimized:**
- **Peak Publishing:** 56% of articles published 0:00-1:00
- **Optimal Schedule:** 1:00-2:00 AM batch processing
- **Implementation:** Cron job `0 1 * * *` for daily pipeline
- **Efficiency:** Single batch vs multiple incremental updates

**Weekend Processing Required:**
- **Weekend News:** 22% of total (172/766 articles)
- **Implication:** 7-day pipeline operation required
- **Schedule:** Daily cron job (Mon-Sun)
- **Storage:** Weekend partitioning strategy

#### NLP Architecture Validation

**Vietnamese Processing Stack Confirmed:**
```yaml
nlp_pipeline:
  tokenization:
    primary: "underthesea"
    purpose: "Word segmentation and POS tagging"
    validation: "Vietnamese content confirmed"
  
  sentiment_analysis:
    primary: "PhoBERT-based models"
    secondary: "VADER multilingual"
    validation: "259 char leads suitable for analysis"
  
  embeddings:
    model: "paraphrase-multilingual-mpnet-base-v2"
    dimensions: 768
    validation: "64 char titles, 259 char leads - no truncation needed"
```

**Content Length Validation:**
- **Title Length:** Mean 64 chars (NLP models typically handle up to 512)
- **Lead Length:** Mean 259 chars (well within model limits)
- **Implication:** No truncation or chunking required
- **Performance:** Direct processing without preprocessing overhead

#### Storage Architecture Validation

**Storage Growth Projections:**
```
Current (2025-2026): 766 articles = 2GB/year
Projection (3x growth): 2,298 articles = 6GB/year
5-year horizon: ~30GB total (within 50GB allocation)
10-year horizon: ~60GB total (requires expansion planning)
```

**Partitioning Strategy Optimized:**
```yaml
partitioning:
  bronze:
    by: ["date"]
    format: "YYYY/MM/DD"
    rationale: "Date-based queries for temporal analysis"
  
  silver:
    by: ["source", "date"]
    format: "source/YYYY/MM/DD"
    rationale: "Source-specific analysis patterns"
  
  gold:
    by: ["year", "month"]
    format: "YYYY/MM"
    rationale: "Monthly aggregations and reporting"
```

**Compression Strategy:**
- **Bronze:** No compression (preserve original)
- **Silver:** Snappy compression (balance speed/size)
- **Gold:** Zstandard compression (optimize storage)
- **Rationale:** Based on 2GB/year current volume

### 15.2 Architecture Updates Based on EDA

#### Processing Architecture Updates

**Refined Pipeline Timing:**
```yaml
pipeline_schedule:
  bronze_import:
    frequency: "daily"
    time: "01:00"  # 1:00 AM
    window: "5 minutes"
    rationale: "Process after midnight publishing peak"
  
  silver_processing:
    frequency: "daily"
    time: "01:10"
    window: "10 minutes"
    dependencies: ["bronze_import"]
  
  gold_features:
    frequency: "daily"
    time: "01:20"
    window: "15 minutes"
    dependencies: ["silver_processing"]
  
  analysis_pipeline:
    frequency: "daily"
    time: "01:35"
    window: "5 minutes"
    dependencies: ["gold_features"]
```

**Weekend Processing Configuration:**
```yaml
weekend_processing:
  enabled: true
  schedule: "daily 7 days/week"
  rationale: "22% of news published on weekends"
  monitoring: "Track weekend vs weekday patterns"
```

#### NLP Architecture Updates

**Vietnamese-Specific Processing:**
```yaml
vietnamese_nlp:
  preprocessing:
    text_normalization: "underthesea.word_tokenize"
    pos_tagging: "underthesea.pos_tag"
    dependency: "UD Vietnamese GSD"
  
  sentiment_analysis:
    model: "vinai/phobert-base-vietnamese-sentiment"
    fallback: "multilingual VADER"
    threshold: "0.5 for binary classification"
  
  entity_recognition:
    stock_tickers: "Pattern matching + VN30 list"
    company_names: "Vietnamese NER models"
    financial_terms: "Domain-specific lexicon"
```

#### Performance Architecture Updates

**Real-time Performance Targets:**
```yaml
performance_targets:
  real_time_processing:
    per_article: "2 seconds"
    batch_100: "30 seconds"
    validation: "<5 minutes for daily volume"
  
  scalability:
    current_volume: "1.4 articles/day"
    target_3x: "4.2 articles/day (<15 min)"
    target_10x: "14 articles/day (<30 min)"
    limit: "50 articles/day (1 hour)"
```

**Quality Targets Updated:**
```yaml
quality_targets:
  current_achievement:
    completeness: "95.8%"
    consistency: "100%"
    accuracy: "High"
  
  thresholds:
    bronze_to_silver: ">80%" (current: 95.8%)
    silver_to_gold: ">90%" (target for implementation)
    analysis_ready: "100%" (requirement)
```

### 15.3 Updated Architecture Decision Records

#### ADR-004: Real-time Processing Strategy
**Date:** 2026-07-11
**Status:** Accepted
**Decision:** Implement real-time processing for current 1.4 articles/day volume
**Rationale:**
- Low volume makes real-time feasible and efficient
- Midnight publishing peak supports batch processing
- EDA validates <5 minute processing time
**Consequences:**
- + Immediate data availability
- + Simplified architecture
- + Better user experience
- - Requires 7-day operation
- - Weekend processing needed

#### ADR-005: Vietnamese NLP Stack
**Date:** 2026-07-11
**Status:** Accepted
**Decision:** Implement Vietnamese-specific NLP stack (underthesea + PhoBERT)
**Rationale:**
- Vietnamese content confirmed in 100% of articles
- 64/259 character lengths suitable for direct processing
- EDA validates approach with existing content
**Consequences:**
- + Optimized for Vietnamese language
- + Better sentiment accuracy
- + No truncation overhead
- - Additional language-specific dependencies
- - Limited multilingual transfer

### 15.4 Implementation Priority Updates

**High Priority (Based on EDA):**
1. **Midnight Batch Processing:** Implement 1:00-2:00 AM pipeline
2. **Vietnamese NLP:** Deploy underthesea + PhoBERT stack
3. **Weekend Processing:** Ensure 7-day operation
4. **Real-time Architecture:** Leverage low-volume capability

**Medium Priority:**
1. **Source Normalization:** Implement SSI/CafeF/VNDirect processing
2. **Temporal Features:** Add midnight publishing pattern features
3. **Quality Monitoring:** Track 95.8% baseline quality

**Low Priority:**
1. **Performance Optimization:** Current <5 min performance adequate
2. **Storage Expansion:** 2GB/year well within 50GB allocation
3. **Scalability Planning:** 3x growth supported by current architecture

---

## 16. EDA Pipeline Architecture (10-Phase Plan)

> Implements `docs/EDA_Guide_Stock_Volatility_Price_News.md`. The lakehouse (Bronze/Silver/Gold) from sections 1–3 remains the data backbone; this section adds the **EDA analysis layer** that consumes Silver/Gold and writes evidence artifacts to `eda_output/`. The defining constraint is **leakage safety** for the downstream volatility-prediction targets (T+1/T+5/T+10 realized volatility).

### 16.1 EDA Module Layout
```
src/eda/
├── phase01_profiling.py        # Phase 1: dataset profiling table
├── phase02_quality.py          # Phase 2: missingness, duplicates, invalid values
├── phase03_price_eda.py        # Phase 3: returns, realized vol, ATR, ACF/PACF, rolling
├── phase04_news_eda.py         # Phase 4: coverage, publish-time, effective_trading_date, topics
├── phase05_relationship.py     # Phase 5: Pearson/Spearman/MI/Granger/cross-corr
├── phase06_event_study.py      # Phase 6: T-10/T-5/T-1 → T+1/T+5/T+10 windows
├── phase07_sparse_news.py      # Phase 7: news_count_1d/3d/5d, days_since_last_news, news_available flag
├── phase08_feature_validation.py  # Phase 8: variance/redundancy/collinearity/drift
├── phase09_leakage.py          # Phase 9: leakage detection + explicit leakage list
├── phase10_visualizations.py   # Phase 10: 11 required charts
└── report.py                   # Final report assembly
```
Each phase is a standalone module: `run_phase(tickers, output_dir) -> artifacts`. They reuse existing loaders (`src/data/load_prices.py`, `load_news.py`) and Silver outputs; they must NOT mutate source data.

### 16.2 Leakage-Safe Target Engineering (critical)
Volatility prediction targets must be computed with strict temporal ordering to prevent look-ahead bias:
```
For each ticker, at trading date t:
    rv_t+h = sqrt( sum(log_return^2) over [t+1, t+h] )   for h in {1, 5, 10}
```
- Targets use ONLY future returns relative to t — never available at prediction time. ✅
- Any feature aligned to date t must use information with timestamp ≤ close of trading day t (or ≤ pub_time for news, mapped via `effective_trading_date`).
- Rolling/ewm features must use `min_periods` and right-aligned windows; `.shift()` is applied before joining with targets to avoid same-row leakage.
- Train/test splits are **time-based** (e.g., train ≤ 2024, test ≥ 2025), never random — enforced in Phase 9.

### 16.3 eda_output/ Structure
```
eda_output/
├── profiling/            # Phase 1: profiling_table.csv
├── quality/              # Phase 2: missingness_*.csv, duplicate_report.json, invalid_values.json
├── price/                # Phase 3: returns_dist.png, rolling_vol.png, acf_pacf.png, corr_heatmap.png
├── news/                 # Phase 4: coverage_*.csv, publish_time.png, sentiment_summary.json
├── relationship/         # Phase 5: corr_matrix.csv, granger_results.json, cross_corr.png
├── feature_engineering/  # Phase 8: feature_report.csv, collinearity.json, drop_recommendations.json
├── leakage/              # Phase 9: leakage_list.md (EXPLICIT), leakage_checks.json
└── report/               # Final report: eda_final_report.md + candidate_features.csv
```

### 16.4 EDA Data Flow
```
Silver (prices_cleaned, news_cleaned) ──┐
Gold (sentiment, volatility)  ─────────┤──▶ src/eda/phaseXX ──▶ eda_output/<phase>/
macro (DXY, SBV rates) ────────────────┘                          │
                                                                  ▼
                                              src/eda/report.py ──▶ eda_output/report/eda_final_report.md
```
- EDA reads Silver/Gold (read-only); never writes back to the lakehouse.
- Each phase is idempotent: re-running overwrites its own `eda_output/<phase>/` only.
- Phases 3–7 must be runnable on the VN30 subset (VCB, FPT, HPG, SSI, MWG) first, then scaled to all 30 via config (no code change).

### 16.5 EDA-Aware Data Schemas (additions)
```
Silver prices: +ticker, returns, log_returns, atr_14, realized_vol_5d/20d   (ATR + realized vol added for Phase 3)
Silver news:   +effective_trading_date, sentiment_score, news_available(1/0) (Phase 4/7 alignment key)
EDA targets:   rv_t+1, rv_t+5, rv_t+10                                      (Phase 3, leakage-safe)
EDA features:  news_count_1d/3d/5d, days_since_last_news, sentiment_lag_0/1/2/3 (Phase 7, shifted)
```

### 16.6 Architecture Decision Records (EDA)

#### ADR-006: EDA-First, Leakage-Safe by Construction
**Date:** 2026-07-11  **Status:** Accepted
**Decision:** Build a dedicated `src/eda/` layer that produces evidence artifacts before any model training; enforce leakage-safe target/feature engineering as a structural invariant (temporal ordering, time-based splits, explicit leakage list).
**Rationale:** The EDA Guide mandates leakage detection and "no future information"; volatility targets are inherently forward-looking and the most leakage-prone artifact.
**Consequences:** + Reproducible, auditable EDA; + Candidate feature set with known leakage status; − Extra pipeline stage before modeling; − Requires disciplined shift/window handling.

#### ADR-007: EDA Consumes Lakehouse, Does Not Modify It
**Date:** 2026-07-11  **Status:** Accepted
**Decision:** `src/eda/` reads Silver/Gold read-only and writes exclusively to `eda_output/`.
**Rationale:** Preserves the Bronze/Silver/Gold immutability contract; keeps EDA experiments isolated and re-runnable.
**Consequences:** + No lakehouse pollution; + Easy to discard/rebuild EDA; − Some recomputation across phases (acceptable for EDA scale).

---

*Document Status: Updated with EDA findings + 10-Phase EDA architecture*
*Last Updated: 2026-07-11*
*Architecture Version: 1.2*
*Next Review: 2026-08-11*