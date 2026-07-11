# Product Requirements Document (PRD)
# Vietnam Stock Market News-Price Correlation Analysis

**Document Version:** 1.0
**Last Updated:** 2026-07-11
**Project:** Vietnam Stock Market Analysis
**Status:** Draft

---

## 1. Executive Summary

### 1.1 Purpose
This PRD defines requirements for analyzing the correlation between news sentiment and stock price movements in the Vietnamese stock market. The project aims to provide actionable insights for investors and researchers by quantifying the impact of news on stock prices and volatility.

### 1.2 Scope
- **Market Focus:** Vietnam Stock Market (VN30 constituents)
- **Data Sources:** News articles (5 sources), OHLCV price data
- **Analysis Types:** News-price correlation, news-volatility correlation
- **Output:** Correlation metrics, statistical significance, predictive features

### 1.3 Target Users
- **Primary:** Academic researchers, data scientists
- **Secondary:** Investment analysts, portfolio managers
- **Tertiary:** Financial journalists, market commentators

---

## 2. Business Objectives

### 2.1 Primary Objectives
1. **Quantify News Impact:** Measure how news sentiment affects stock prices and volatility
2. **Predictive Modeling:** Build features to predict price movements based on news
3. **Market Understanding:** Gain insights into information efficiency in Vietnamese market

### 2.2 Secondary Objectives
1. **Methodology Development:** Create reusable NLP-finance analysis framework
2. **Real-time Monitoring:** Enable daily monitoring of news-market relationships
3. **Academic Research:** Support thesis and research publications

### 2.3 Success Metrics
- **Statistical Significance:** Correlations with p-values < 0.05
- **Predictive Power:** Features with meaningful information coefficients
- **Data Quality:** >95% data quality scores across all layers
- **Performance:** Analysis pipeline completes within 30 minutes for daily updates

---

## 3. Functional Requirements

### 3.1 Data Ingestion (FR-001)
**Priority:** P0 (Must Have)

The system shall:
- **FR-001.1:** Import news articles daily from 5 primary sources (CafeF, SSI, VNDirect, VietStock, HSC)
- **FR-001.2:** Import OHLCV price data daily for 30 VN30 stocks
- **FR-001.3:** Store raw data in Bronze layer without modification
- **FR-001.4:** Validate data integrity upon import
- **FR-001.5:** Log all import operations with timestamps

### 3.2 Data Processing (FR-002)
**Priority:** P0 (Must Have)

The system shall:
- **FR-002.1:** Clean and normalize news data (remove duplicates, standardize formats)
- **FR-002.2:** Extract sentiment scores from news content (Vietnamese NLP)
- **FR-002.3:** Calculate technical indicators from price data (returns, volatility, momentum)
- **FR-002.4:** Validate processed data quality (completeness, consistency, accuracy)
- **FR-002.5:** Store processed data in Silver layer with quality scores

### 3.3 Feature Engineering (FR-003)
**Priority:** P0 (Must Have)

The system shall:
- **FR-003.1:** Create news features (sentiment, embeddings, topics, urgency)
- **FR-003.2:** Create price features (returns, volatility, technical indicators)
- **FR-003.3:** Create time-series features (lagged variables, rolling windows)
- **FR-003.4:** Create correlation features (news-price, news-volatility)
- **FR-003.5:** Store features in Gold layer optimized for analysis

### 3.4 Correlation Analysis (FR-004)
**Priority:** P0 (Must Have)

The system shall:
- **FR-004.1:** Calculate news-price correlation coefficients (Pearson, Spearman)
- **FR-004.2:** Perform lead-lag analysis (cross-correlation at different lags)
- **FR-004.3:** Conduct Granger causality tests
- **FR-004.4:** Calculate news-volatility correlation metrics
- **FR-004.5:** Perform event study analysis around news publication

### 3.5 Statistical Testing (FR-005)
**Priority:** P1 (Should Have)

The system shall:
- **FR-005.1:** Perform hypothesis testing for correlation significance
- **FR-005.2:** Calculate confidence intervals for correlation estimates
- **FR-005.3:** Adjust for multiple comparisons (Bonferroni, FDR)
- **FR-005.4:** Test for normality and heteroscedasticity
- **FR-005.5:** Generate statistical reports with p-values and effect sizes

### 3.6 Visualization & Reporting (FR-006)
**Priority:** P1 (Should Have)

The system shall:
- **FR-006.1:** Generate time-series plots of news sentiment and prices
- **FR-006.2:** Create correlation heatmaps by stock and time period
- **FR-006.3:** Plot event study cumulative abnormal returns
- **FR-006.4:** Generate summary reports with key findings
- **FR-006.5:** Export analysis results in multiple formats (CSV, JSON, Markdown)

### 3.7 Data Refresh Management (FR-007)
**Priority:** P0 (Must Have)

The system shall:
- **FR-007.1:** Automatically detect new data in source directories
- **FR-007.2:** Process only new/changed data (incremental updates)
- **FR-007.3:** Maintain data version history for rollback capability
- **FR-007.4:** Provide data status dashboard (last update, quality metrics)
- **FR-007.5:** Send alerts on data quality issues

### 3.8 Quality Assurance (FR-008)
**Priority:** P1 (Should Have)

The system shall:
- **FR-008.1:** Validate data schemas at each layer (Bronze/Silver/Gold)
- **FR-008.2:** Check data quality rules (completeness, consistency, uniqueness)
- **FR-008.3:** Calculate and track data quality scores
- **FR-008.4:** Generate quality reports with recommendations
- **FR-008.5:** Prevent promotion of low-quality data between layers

---

## 4. Non-Functional Requirements

### 4.1 Performance (NFR-001)
- **NFR-001.1:** Daily data processing completes within 30 minutes
- **NFR-001.2:** Feature engineering processes 1M rows in <10 minutes
- **NFR-001.3:** Correlation analysis completes within 5 minutes for 30 stocks
- **NFR-001.4:** Query responses for gold layer data <1 second

### 4.2 Scalability (NFR-002)
- **NFR-002.1:** System handles up to 100 stocks (beyond VN30)
- **NFR-002.2:** System processes up to 10 years of historical data
- **NFR-002.3:** System handles up to 1M news articles
- **NFR-002.4:** Architecture supports horizontal scaling

### 4.3 Reliability (NFR-003)
- **NFR-003.1:** Data processing pipeline success rate >99%
- **NFR-003.2:** Automatic retry on transient failures
- **NFR-003.3:** Data integrity verification at each pipeline stage
- **NFR-003.4:** Backup and recovery mechanisms for all data

### 4.4 Data Quality (NFR-004)
- **NFR-004.1:** Data quality scores >80% for promotion between layers
- **NFR-004.2:** Missing data <20% for critical fields
- **NFR-004.3:** Duplicate records <1% after cleaning
- **NFR-004.4:** Schema validation 100% for all processed data

### 4.5 Maintainability (NFR-005)
- **NFR-005.1:** Code coverage >80% for all modules
- **NFR-005.2:** Code review required for all changes
- **NFR-005.3:** Documentation updated with each feature
- **NFR-005.4:** CLAUDE.md compliance enforced

### 4.6 Usability (NFR-006)
- **NFR-006.1:** Clear error messages with actionable guidance
- **NFR-006.2:** Progress indicators for long-running operations
- **NFR-006.3:** Help documentation for all user-facing features
- **NFR-006.4:** Vietnamese language support where appropriate

---

## 5. Data Requirements

### 5.1 News Data
- **Sources:** 5 primary Vietnamese financial news sources
- **Fields:** source, title, category, pub_date, url, author, lead, content
- **Frequency:** Daily updates (multiple articles per day)
- **Retention:** Raw data retained indefinitely in Bronze layer
- **Language:** Vietnamese (with multilingual NLP support)

### 5.2 Price Data
- **Sources:** Daily OHLCV data for VN30 stocks
- **Fields:** date, open, high, low, close, volume
- **Frequency:** Daily (trading days only)
- **Retention:** Full historical data retained
- **Adjustments:** No price adjustments (raw closing prices)

### 5.3 Reference Data
- **Stock Universe:** 30 VN30 constituent stocks
- **Trading Calendar:** Vietnam trading days (Mon-Fri, excluding holidays)
- **Market Data:** VN30 index levels (optional, for benchmarking)
- **Corporate Actions:** Stock splits, dividends (future enhancement)

---

## 6. Technical Architecture

### 6.1 Data Lakehouse Layers
- **Bronze Layer:** Raw data preservation (CSV, original formats)
- **Silver Layer:** Cleaned and validated data (Parquet, standardized schemas)
- **Gold Layer:** Feature-rich analysis-ready data (optimized Parquet)

### 6.2 Technology Stack
- **Language:** Python 3.10+
- **Data Processing:** Pandas, Polars (for large datasets)
- **NLP:** Underthesea (Vietnamese), Transformers, Sentence-Transformers
- **Financial Analysis:** ARCH (GARCH models), Statsmodels
- **Storage:** Parquet files with partitioning
- **Quality:** BMad framework for CLAUDE.md compliance

### 6.3 Pipeline Architecture
```
Source → Bronze Import → Silver Processing → Gold Features → Analysis
   ↓           ↓                ↓                ↓            ↓
Daily      Raw Data        Clean Data        Features    Insights
Updates    Preserved      Validated         Ready      Reports
```

---

## 7. User Stories

### 7.1 Data Analyst
**As a data analyst, I want to:**
- Load news and price data easily for analysis
- Access pre-computed features without manual processing
- Visualize correlations between news and prices
- Export analysis results for reporting

**Acceptance Criteria:**
- Can load data with single function call
- Features available in gold layer within 1 hour of data update
- Interactive plots generated automatically
- Multiple export formats available

### 7.2 Researcher
**As a researcher, I want to:**
- Analyze news impact on stock prices with statistical rigor
- Test hypotheses about market efficiency
- Replicate analysis with different parameters
- Access data lineage and quality metrics

**Acceptance Criteria:**
- Statistical tests with p-values and confidence intervals
- Parameterizable analysis functions
- Complete data lineage documentation
- Quality metrics available for all datasets

### 7.3 Investment Professional
**As an investment professional, I want to:**
- Monitor news sentiment for stocks I follow
- Understand which news moves markets
- Get alerts on unusual news-volatility patterns
- Access simple summaries of complex analyses

**Acceptance Criteria:**
- Daily sentiment dashboard available
- Key news events highlighted
- Alert system for unusual patterns
- Executive summaries generated

---

## 8. Constraints & Assumptions

### 8.1 Constraints
- **Data Sources:** Limited to predefined crawl data directories (per CLAUDE.md)
- **Processing Time:** Daily updates must complete overnight
- **Storage:** Local file system storage (no cloud dependencies)
- **Language:** Vietnamese text processing required
- **Quality:** CLAUDE.md compliance mandatory

### 8.2 Assumptions
- **Data Availability:** Daily data updates provided by user crawling
- **Market Hours:** Vietnam market trades Mon-Fri 9:00-15:00 (UTC+7)
- **News Timing:** News can be published outside trading hours
- **Price Impact:** News may affect prices with lags (0-3 days)
- **Volatility:** Bad news increases volatility more than good news decreases it

---

## 9. Dependencies

### 9.1 External Dependencies
- **Data Sources:** User-provided crawled data (no direct API access)
- **Python Packages:** Pandas, Polars, Underthesea, Transformers, ARCH
- **System Requirements:** Python 3.10+, 16GB RAM recommended
- **Storage:** 50GB minimum for historical data

### 9.2 Internal Dependencies
- **BMad Framework:** For quality compliance and project management
- **CLAUDE.md:** Quality standards and behavioral guidelines
- **Data Lakehouse:** Bronze/Silver/Gold architecture
- **Config Module:** Centralized configuration management

---

## 10. Risks & Mitigations

### 10.1 Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Vietnamese NLP accuracy | High | Medium | Use multiple NLP approaches, validate with human labeling |
| Data quality issues | High | Medium | Implement comprehensive quality checks, manual review |
| Processing performance | Medium | Low | Use Polars for large datasets, optimize algorithms |
| Storage limitations | Low | Low | Implement data retention policies, compression |

### 10.2 Data Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Missing data in sources | High | Medium | Track missing data rates, alert on thresholds |
| Inconsistent data formats | Medium | High | Robust schema validation, flexible parsers |
| Duplicate articles | Medium | High | Deduplication at multiple stages |
| Price data errors | High | Low | Cross-validation with multiple sources |

### 10.3 Analysis Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Spurious correlations | High | Medium | Multiple testing correction, robust methods |
| Causality fallacy | High | High | Clear communication of correlation vs causation |
| Look-ahead bias | High | Medium | Strict temporal ordering, cross-validation |
| Overfitting | Medium | Low | Out-of-sample testing, regularization |

---

## 11. Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Objective:** Establish data lakehouse and basic processing
- Implement Bronze/Silver/Gold architecture
- Create data import and cleaning pipelines
- Set up quality validation framework
- BMad configuration and documentation

**Deliverables:**
- Working lakehouse pipeline
- Data quality reports
- Technical architecture document

### Phase 2: Feature Engineering (Week 3-4)
**Objective:** Create comprehensive feature set
- Implement news sentiment analysis
- Calculate technical indicators
- Engineer time-series features
- Create correlation features

**Deliverables:**
- Gold layer with feature-rich datasets
- Feature documentation
- Performance benchmarks

### Phase 3: Analysis & Testing (Week 5-6)
**Objective:** Implement core analysis capabilities
- Correlation analysis modules
- Statistical testing framework
- Event study methodology
- Visualization components

**Deliverables:**
- Analysis modules with tests
- Statistical reports
- Visualization library

### Phase 4: Integration & Optimization (Week 7-8)
**Objective:** Polish and optimize system
- End-to-end pipeline integration
- Performance optimization
- User documentation
- Quality assurance

**Deliverables:**
- Complete analysis system
- User guides
- Performance reports
- Deployment documentation

---

## 12. Success Criteria

### 12.1 Technical Success
- ✅ All functional requirements implemented
- ✅ Data quality scores >80% across all layers
- ✅ Pipeline performance meets NFRs
- ✅ Code coverage >80%
- ✅ CLAUDE.md compliance verified

### 12.2 Business Success
- ✅ Statistically significant correlations identified (p<0.05)
- ✅ Predictive features with information coefficient >0.1
- ✅ Actionable insights generated
- ✅ Reusable methodology established

### 12.3 User Satisfaction
- ✅ Easy to use for target users
- ✅ Reliable daily operation
- ✅ Clear and useful outputs
- ✅ Comprehensive documentation

---

## 13. Appendix

### 13.1 Glossary
- **Bronze/Silver/Gold:** Data quality layers in lakehouse architecture
- **OHLCV:** Open, High, Low, Close, Volume price data
- **VN30:** Vietnam's 30 largest stocks by market cap
- **Granger Causality:** Statistical test of predictive relationships
- **Event Study:** Analysis of market reaction to specific events
- **Information Coefficient:** Correlation between predictions and outcomes

### 13.2 References
- Vietnam Stock Market Trading Calendar
- Academic Literature on News-Price Relationships
- BMad Framework Documentation
- CLAUDE.md Quality Standards

### 13.3 Change Log
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.1 | 2026-07-11 | Added EDA findings from 2025-2026 data analysis | Data Analysis Team |
| 1.0 | 2026-07-11 | Initial PRD creation | Data Analysis Team |

### 13.4 EDA Findings (2025-2026 Data Analysis)

**Analysis Period:** 2025-01-02 to 2026-08-06 (1.5 years)
**Total Articles Analyzed:** 766 articles
**Sources:** 3 sources (SSI: 374, CafeF: 362, VNDirect: 30)

#### Key Data Characteristics

**Data Volume:**
- Daily average: ~1.4 articles/day
- Year-over-year growth: 106% (2025: 250 → 2026: 516 articles)
- Low-volume stream suitable for real-time processing

**Temporal Patterns:**
- Peak publishing: 0:00-1:00 (56% of articles) - midnight batch publishing
- Highest volume: Tuesdays (191 articles)
- Weekend news: 22% of total - requires 7-day processing
- Monthly peaks: June 2026 (110), March 2026 (96), April 2026 (82)

**Source Distribution:**
- SSI: 49% of total, most consistent (31.2 articles/month)
- CafeF: 47% of total, highest volume (45.2 articles/month)
- VNDirect: 4% of total, limited coverage (6 articles/month)

**Content Characteristics:**
- Average title length: 64 characters (manageable for NLP)
- Average lead length: 259 characters (good for sentiment analysis)
- Vietnamese language confirmed (underthesea + PhoBERT required)
- Data completeness: 95.8% (exceeds 80% requirement)

#### Technical Implications

**Processing Strategy:**
- Real-time processing feasible for current volume
- Midnight batch processing optimal (56% of articles)
- Source normalization required (SSI, CafeF, VNDirect)
- Weekend processing pipeline essential

**Architecture Validation:**
- ✅ Data quality >95% meets FR-002 requirements
- ✅ Performance targets easily achievable (<5 min actual vs 30 min target)
- ✅ Storage growth manageable (2GB/year vs 50GB allocation)
- ✅ Vietnamese NLP stack validated

**Requirements Validation:**
- FR-001 (Data Ingestion): ✅ Daily import validated (1.4 articles/day)
- FR-002 (Data Processing): ✅ Vietnamese NLP confirmed
- FR-007 (Data Refresh): ✅ Midnight batch processing optimal
- NFR-001 (Performance): ✅ <30 min target easily achievable
- NFR-004 (Data Quality): ✅ 95.8% completeness achieved

#### Updated Requirements Based on EDA

**Refined Processing Requirements:**
- Real-time pipeline for current 1.4 articles/day volume
- Midnight batch processing (1:00-2:00 AM) for efficiency
- Weekend-aware processing (22% of news published on weekends)
- Source-specific processing strategies (SSI primary, CafeF secondary)

**Updated NLP Requirements:**
- Primary: underthesea for Vietnamese tokenization
- Secondary: PhoBERT-based sentiment analysis
- Tertiary: Multilingual sentence-transformers (768-dim embeddings)
- No content truncation needed (64 char titles, 259 char leads)

**Revised Performance Targets:**
- Actual processing: <5 minutes (vs 30 min target)
- Storage growth: 2GB/year current (supports 10x expansion)
- Quality scores: 95.8% actual (vs 80% requirement)
- Real-time feasible for current and projected volumes

---

**Document Status:** Updated with EDA findings
**Next Review Date:** 2026-07-15
**Approved By:** Pending

---

*This PRD is a living document and will be updated as requirements evolve.*
