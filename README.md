# 🏏 IPL Natural Language Query System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.29.0-orange.svg)](https://gradio.app/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

An intelligent cricket statistics explorer that allows users to query IPL (Indian Premier League) data using natural language. Powered by advanced language models and LangGraph workflows, this system converts plain English questions into SQL queries and provides comprehensive answers about IPL cricket statistics.

## 🌟 Features

- **Natural Language Interface**: Ask questions in plain English about IPL statistics
- **Multi-Model Support**: Choose from multiple LLMs including Gemini 2.0 Flash, Llama 3.3, and Gemini 2.5 Flash
- **Intelligent Query Processing**: Automatically expands vague questions into detailed, unambiguous queries
- **Real-time SQL Generation**: Converts natural language to optimized PostgreSQL queries
- **Interactive Web Interface**: Clean, responsive Gradio-based UI with real-time results
- **Comprehensive Database**: Complete IPL data including players, teams, matches, deliveries, and venues
- **Query Transparency**: Shows the expanded question, generated SQL query, and raw results

## 🏗️ Architecture

The system uses a sophisticated workflow built with LangGraph:

1. **Natural Language Expansion**: Clarifies and expands user questions
2. **SQL Query Generation**: Converts expanded questions to PostgreSQL queries
3. **Query Execution**: Runs queries against the IPL database
4. **Answer Generation**: Formats results into user-friendly responses

## 📊 Database Schema

The application works with a comprehensive IPL database containing:

- **Players**: All IPL players with detailed information
- **Teams**: Complete IPL franchise data
- **Matches**: Match-by-match records with metadata
- **Player Matches**: Individual player statistics per match
- **Deliveries**: Ball-by-ball data for every match
- **Venues**: All IPL venues
- **Player Teams**: Player-team relationships

![Database Schema](ipl-database-erd.png)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database with IPL data
- API keys for supported LLM providers

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ItzManan/ipl-nlp.git
   cd ipl-nlp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   DATABASE_URL=postgresql://username:password@localhost:5432/ipl-database
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the interface**:
   Open your browser and navigate to the local URL shown in the terminal (typically `http://127.0.0.1:7860`)

## 💻 Usage Examples

Ask questions like:

- **Player Statistics**: "How many sixes did Virat Kohli hit in IPL 2023?"
- **Team Performance**: "Which team has the highest average score batting first?"
- **Match Analysis**: "Top 5 bowlers with best economy rate in powerplay overs"
- **Historical Data**: "Most centuries scored by a player in IPL history"
- **Comparative Analysis**: "MS Dhoni vs Virat Kohli head-to-head stats"

## 🔧 Configuration

### Supported LLM Models

- **Gemini 2.0 Flash**: Google's latest conversational AI model
- **Llama 3.3 70B Versatile**: Meta's large language model via Groq
- **Gemini 2.5 Flash Preview**: Google's experimental model

### Database Configuration

The system expects a PostgreSQL database with the following key tables:
- `players`, `teams`, `matches`, `player_matches`, `deliveries`, `venues`, `player_teams`

## 🛠️ Technical Stack

- **Backend**: Python with LangChain and LangGraph
- **Frontend**: Gradio for interactive web interface
- **Database**: PostgreSQL with psycopg2-binary
- **LLM Integration**: Google Generative AI, Groq
- **Environment Management**: python-dotenv

## 🔍 How It Works

1. **User Input**: Natural language question about IPL statistics
2. **Question Expansion**: LLM clarifies and expands the question for better SQL generation
3. **SQL Generation**: Advanced prompt engineering converts the expanded question to PostgreSQL
4. **Query Execution**: SQL query runs against the IPL database
5. **Result Processing**: Raw results are formatted into a user-friendly markdown response

## 🎯 IPL-Specific Features

The system includes cricket-specific logic for:

- **Batting Order Analysis**: Distinguishing between batting first and chasing teams
- **Match Context**: Understanding winning/losing team perspectives
- **Player-Team Relationships**: Accurate mapping across seasons
- **Season-Specific Queries**: Filtering by IPL seasons
- **Cricket Terminology**: Proper handling of overs, balls, runs, wickets, etc.

## 📈 Data Coverage

- **Timeline**: Complete IPL data up to May 6, 2025
- **Granularity**: Ball-by-ball data for comprehensive analysis
- **Scope**: All teams, players, matches, and venues
- **Statistics**: Batting, bowling, fielding, and team performance metrics
---

**Note**: This application requires valid API keys for the supported LLM providers and a properly configured PostgreSQL database with IPL data.