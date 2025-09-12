# MCP Pinecone Server

## Overview
This is a FastMCP (Model Context Protocol) server that provides semantic search capabilities using Pinecone vector database and OpenAI embeddings. The server exposes two main tools for searching and retrieving data from a Pinecone index.

## Purpose
- Provides semantic search functionality over vector embeddings
- Integrates OpenAI embedding models with Pinecone vector database
- Offers MCP-compatible API for use with AI assistants and applications

## Current State
- ✅ Python 3.11 environment configured
- ✅ All dependencies installed (fastmcp, openai, pinecone-client)
- ✅ API keys configured (OPENAI_API_KEY, PINECONE_API_KEY)
- ✅ MCP server running on port 8000
- ✅ Deployment configured for VM (always-on server)

## Recent Changes (September 8, 2025)
- Initial project setup in Replit environment
- Installed Python dependencies via package manager
- Configured API keys for OpenAI and Pinecone
- Set up workflow to run MCP server
- Configured deployment for production as VM service

## Project Architecture
- **Language**: Python 3.11
- **Main Server**: `mcp-poc-server.py` - FastMCP server with Pinecone integration
- **Dependencies**: fastmcp, openai, pinecone-client
- **Environment**: 
  - OPENAI_API_KEY: For generating query embeddings
  - PINECONE_API_KEY: For vector database access
  - PINECONE_INDEX: poc-discovered-data-query (default)
  - PINECONE_NAMESPACE: poc (default)
  - EMBED_MODEL: text-embedding-3-large (default)
  - INDEX_DIM: 3072 (default)

## Server Features
- **Search Tool**: Semantic search over Pinecone with optional resource type filtering
- **Fetch Tool**: Retrieve specific items by ID from Pinecone
- **HTTP Transport**: Runs on http://127.0.0.1:8000/mcp
- **Stateless**: Configured for stateless HTTP operation

## Deployment
- **Type**: VM (always-on)
- **Command**: `python mcp-poc-server.py`
- **Port**: 8000
- **Transport**: HTTP with MCP protocol