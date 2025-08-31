# AAOIFI Islamic Finance Knowledge System

## Overview
An AI-powered Islamic finance knowledge system that processes AAOIFI Shari'ah standards to generate verified question-answer datasets with strict reference validation. The project focuses on creating high-quality bilingual (Arabic/English) judgmental datasets for training AI models on Islamic finance compliance.

## Project Architecture

### Core Components
1. **Data Processing Pipeline**: Processes AAOIFI Shari'ah Standards in Arabic and English
2. **Reference Verification System**: Ensures all generated content is backed by authentic AAOIFI sources
3. **Judgmental Dataset Generator**: Creates True/False verification examples with strict validation
4. **Quality Control**: Multi-stage validation with human review sampling

### Key Features
- Bilingual support (Arabic/English)
- Strict reference validation against canonical AAOIFI texts
- API key rotation and quota management
- Checkpointing and resume capability
- Human-in-the-loop quality assurance

## File Structure
```
inputs/                     # Canonical AAOIFI data
├── arabic_cleaned.txt      # Cleaned Arabic text
├── english_cleaned.txt     # Cleaned English text
├── arabic_chunks.json      # Chunked Arabic content
├── english_chunks.json     # Chunked English content
├── arabic_qa_pairs.json    # Arabic Q&A seeds
└── english_qa_pairs.json   # English Q&A seeds

data/generation_stage_B/    # Output datasets
├── ar/                     # Arabic generation
├── en/                     # English generation
└── merged/                 # Combined datasets

scripts/                    # Validation and utilities
progress/                   # State management
logs/                      # Processing logs
raw_responses/             # Raw API responses
```

## User Preferences
- Prioritize data integrity over speed
- Never fabricate references - use "UNKNOWN" when uncertain
- Generate Arabic examples first, then English
- Maintain strict 50/50 True/False balance
- Implement comprehensive logging and checkpointing

## Technical Decisions
- Use Gemini 2.5 Pro/Flash models with key rotation
- Implement exponential backoff for API errors
- Store raw responses for debugging
- Generate minimum 2000 examples per language
- 5% human review sampling for quality assurance

## Recent Changes
- 2025-01-26: Initial project setup with AAOIFI dataset generation requirements
- API keys configured for Gemini models with rotation system
- Smoke test validation system implemented