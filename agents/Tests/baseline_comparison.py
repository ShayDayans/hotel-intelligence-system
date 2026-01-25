"""
Baseline Comparison Tests

Compares our Hotel Intelligence System against:
1. Raw Data Queries - Direct database search without LLM
2. Simple Chatbot - LLM without tools/RAG
3. Our System - Full agent pipeline

Measures:
- Response time
- Response quality (manual rating or LLM-as-judge)
- Data coverage (sources used)
- Actionability of insights
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

# Add paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _AGENTS_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))


# ===========================================
# DATA CLASSES
# ===========================================

@dataclass
class ComparisonResult:
    """Result from a single comparison test."""
    query: str
    system_type: str  # "raw_data", "simple_chatbot", "full_system"
    response: str
    elapsed_seconds: float
    sources_used: list = field(default_factory=list)
    tools_called: list = field(default_factory=list)
    error: Optional[str] = None
    
    # Quality metrics (to be filled manually or by LLM judge)
    relevance_score: Optional[float] = None  # 1-5
    actionability_score: Optional[float] = None  # 1-5
    accuracy_score: Optional[float] = None  # 1-5
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonSummary:
    """Summary of comparison across all systems."""
    query: str
    results: list  # List of ComparisonResult
    winner: Optional[str] = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ===========================================
# BASELINE 1: RAW DATA QUERIES
# ===========================================

class RawDataBaseline:
    """
    Baseline: Direct database queries without LLM processing.
    Simulates what a user would get from raw data.
    """
    
    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_pinecone import PineconeVectorStore
        
        print("[RawData] Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
        self.index_name = 'booking-agent'
    
    def _get_vectorstore(self, namespace: str):
        from langchain_pinecone import PineconeVectorStore
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace
        )
    
    def query(self, query: str, hotel_id: str = None, k: int = 5) -> ComparisonResult:
        """
        Execute a raw data query - just returns matching documents.
        No LLM processing, no synthesis.
        """
        start_time = time.time()
        sources_used = []
        all_results = []
        
        try:
            # Search across all namespaces
            namespaces = ['booking_reviews', 'airbnb_reviews', 'booking_hotels', 'airbnb_hotels']
            
            for namespace in namespaces:
                try:
                    vs = self._get_vectorstore(namespace)
                    filter_dict = {"hotel_id": hotel_id} if hotel_id else None
                    docs = vs.similarity_search(query, k=k, filter=filter_dict)
                    
                    if docs:
                        sources_used.append(namespace)
                        for doc in docs:
                            all_results.append({
                                "source": namespace,
                                "content": doc.page_content[:300],
                                "metadata": doc.metadata
                            })
                except Exception as e:
                    print(f"[RawData] Error searching {namespace}: {e}")
            
            # Format as raw output (no synthesis)
            if all_results:
                response = f"=== Raw Data Results for: '{query}' ===\n\n"
                response += f"Found {len(all_results)} matching documents from {len(sources_used)} sources.\n\n"
                
                for i, result in enumerate(all_results[:10], 1):
                    response += f"[{i}] Source: {result['source']}\n"
                    response += f"    Hotel: {result['metadata'].get('hotel_id', 'N/A')}\n"
                    response += f"    Content: {result['content']}...\n\n"
            else:
                response = f"No results found for query: '{query}'"
            
            elapsed = time.time() - start_time
            
            return ComparisonResult(
                query=query,
                system_type="raw_data",
                response=response,
                elapsed_seconds=elapsed,
                sources_used=sources_used,
                tools_called=["similarity_search"]
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return ComparisonResult(
                query=query,
                system_type="raw_data",
                response="",
                elapsed_seconds=elapsed,
                error=str(e)
            )


# ===========================================
# BASELINE 2: SIMPLE CHATBOT (LLM only)
# ===========================================

class SimpleChatbotBaseline:
    """
    Baseline: LLM without tools or RAG.
    Just asks the LLM directly - simulates a generic chatbot.
    """
    
    def __init__(self):
        from base_agent import LLMWithFallback
        self.llm = LLMWithFallback()
    
    def query(self, query: str, hotel_name: str = "the hotel") -> ComparisonResult:
        """
        Execute a simple chatbot query - LLM only, no tools.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        start_time = time.time()
        
        try:
            # Simple system prompt (no tools, no data access)
            system_prompt = f"""You are a helpful hotel assistant. 
You do NOT have access to any real data or reviews about specific hotels.
You can only provide general information based on your training data.
If asked about specific reviews or data, acknowledge you don't have access to real-time information.

The user is asking about: {hotel_name}"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            response = self.llm.invoke(messages)
            elapsed = time.time() - start_time
            
            return ComparisonResult(
                query=query,
                system_type="simple_chatbot",
                response=response.content,
                elapsed_seconds=elapsed,
                sources_used=["LLM training data only"],
                tools_called=[]
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return ComparisonResult(
                query=query,
                system_type="simple_chatbot",
                response="",
                elapsed_seconds=elapsed,
                error=str(e)
            )


# ===========================================
# OUR SYSTEM: FULL AGENT PIPELINE
# ===========================================

class FullSystemBaseline:
    """
    Our full system with LangGraph coordinator, agents, tools, and RAG.
    """
    
    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        from coordinator import LangGraphCoordinator
        
        self.coordinator = LangGraphCoordinator(hotel_id, hotel_name, city)
        self.state = self.coordinator.get_initial_state()
        self.hotel_name = hotel_name
    
    def query(self, query: str) -> ComparisonResult:
        """
        Execute a full system query through the coordinator.
        """
        start_time = time.time()
        
        try:
            response, self.state = self.coordinator.run(query, self.state)
            elapsed = time.time() - start_time
            
            # Extract which agents/tools were used
            agents_used = self.state.get("agents_executed", [])
            
            # Determine sources based on agents
            sources = []
            if "review_analyst" in agents_used:
                sources.extend(["booking_reviews", "airbnb_reviews", "google_maps", "tripadvisor"])
            if "competitor_analyst" in agents_used:
                sources.extend(["booking_hotels", "airbnb_hotels", "nlp_analysis"])
            if "benchmark_agent" in agents_used:
                sources.extend(["lr_analysis", "feature_impact"])
            if "market_intel" in agents_used:
                sources.extend(["events", "weather", "google_maps"])
            
            return ComparisonResult(
                query=query,
                system_type="full_system",
                response=response,
                elapsed_seconds=elapsed,
                sources_used=list(set(sources)),
                tools_called=agents_used
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return ComparisonResult(
                query=query,
                system_type="full_system",
                response="",
                elapsed_seconds=elapsed,
                error=str(e)
            )


# ===========================================
# LLM-AS-JUDGE EVALUATION
# ===========================================

def evaluate_with_llm_judge(results: list[ComparisonResult], query: str) -> dict:
    """
    Use LLM to evaluate and compare responses.
    Returns scores for each system.
    """
    from base_agent import LLMWithFallback
    from langchain_core.messages import HumanMessage
    
    llm = LLMWithFallback()
    
    # Build comparison prompt
    prompt = f"""You are evaluating hotel assistant responses to the query: "{query}"

Rate each response on three criteria (1-5 scale):
1. RELEVANCE: Does it answer the question?
2. ACTIONABILITY: Does it provide specific, useful recommendations?
3. ACCURACY: Does it use real data (not generic/hallucinated)?

Responses to evaluate:

"""
    
    for i, result in enumerate(results, 1):
        prompt += f"--- SYSTEM {i}: {result.system_type.upper()} ---\n"
        prompt += f"Response time: {result.elapsed_seconds:.2f}s\n"
        prompt += f"Sources used: {result.sources_used}\n"
        prompt += f"Response:\n{result.response[:1000]}...\n\n"
    
    prompt += """
Rate each system. Return JSON only:
{
    "raw_data": {"relevance": X, "actionability": X, "accuracy": X, "notes": "..."},
    "simple_chatbot": {"relevance": X, "actionability": X, "accuracy": X, "notes": "..."},
    "full_system": {"relevance": X, "actionability": X, "accuracy": X, "notes": "..."},
    "winner": "system_name",
    "explanation": "..."
}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse JSON from response
        import re
        content = response.content
        
        # Handle markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            content = match.group(1) if match else content
        
        evaluation = json.loads(content)
        return evaluation
        
    except Exception as e:
        print(f"[LLM Judge] Evaluation failed: {e}")
        return {"error": str(e)}


# ===========================================
# COMPARISON RUNNER
# ===========================================

class BaselineComparison:
    """
    Main class to run baseline comparisons.
    """
    
    TEST_QUERIES = [
        {
            "query": "What are guests saying about wifi quality?",
            "type": "review_analysis",
            "expected_agent": "review_analyst"
        },
        {
            "query": "How does my rating compare to competitors?",
            "type": "competitor_analysis",
            "expected_agent": "competitor_analyst"
        },
        {
            "query": "What are my main weaknesses based on reviews?",
            "type": "sentiment_analysis",
            "expected_agent": "review_analyst"
        },
        {
            "query": "Are there any events happening this weekend?",
            "type": "market_intel",
            "expected_agent": "market_intel"
        },
        {
            "query": "Rank hotels by rating in my city",
            "type": "benchmark",
            "expected_agent": "benchmark_agent"
        }
    ]
    
    def __init__(self, hotel_id: str, hotel_name: str, city: str):
        self.hotel_id = hotel_id
        self.hotel_name = hotel_name
        self.city = city
        
        # Initialize baselines
        print("\n" + "="*60)
        print("INITIALIZING BASELINE SYSTEMS")
        print("="*60)
        
        print("\n[1/3] Initializing Raw Data Baseline...")
        self.raw_data = RawDataBaseline()
        
        print("\n[2/3] Initializing Simple Chatbot Baseline...")
        self.simple_chatbot = SimpleChatbotBaseline()
        
        print("\n[3/3] Initializing Full System...")
        self.full_system = FullSystemBaseline(hotel_id, hotel_name, city)
        
        print("\n✅ All baselines initialized")
    
    def run_single_comparison(self, query: str, use_llm_judge: bool = True) -> ComparisonSummary:
        """
        Run a single query through all three systems and compare.
        """
        print(f"\n{'='*60}")
        print(f"COMPARING: {query}")
        print('='*60)
        
        results = []
        
        # 1. Raw Data
        print("\n[1/3] Running Raw Data Baseline...")
        raw_result = self.raw_data.query(query, hotel_id=self.hotel_id)
        results.append(raw_result)
        print(f"   ✓ Completed in {raw_result.elapsed_seconds:.2f}s")
        
        # 2. Simple Chatbot
        print("\n[2/3] Running Simple Chatbot Baseline...")
        chatbot_result = self.simple_chatbot.query(query, hotel_name=self.hotel_name)
        results.append(chatbot_result)
        print(f"   ✓ Completed in {chatbot_result.elapsed_seconds:.2f}s")
        
        # 3. Full System
        print("\n[3/3] Running Full System...")
        full_result = self.full_system.query(query)
        results.append(full_result)
        print(f"   ✓ Completed in {full_result.elapsed_seconds:.2f}s")
        
        # Evaluate with LLM judge
        winner = None
        notes = ""
        
        if use_llm_judge:
            print("\n[4/4] Running LLM Judge Evaluation...")
            evaluation = evaluate_with_llm_judge(results, query)
            
            if "error" not in evaluation:
                winner = evaluation.get("winner")
                notes = evaluation.get("explanation", "")
                
                # Update scores in results
                for result in results:
                    sys_scores = evaluation.get(result.system_type, {})
                    result.relevance_score = sys_scores.get("relevance")
                    result.actionability_score = sys_scores.get("actionability")
                    result.accuracy_score = sys_scores.get("accuracy")
                
                print(f"   ✓ Winner: {winner}")
            else:
                print(f"   ⚠ Evaluation failed: {evaluation['error']}")
        
        return ComparisonSummary(
            query=query,
            results=[r.to_dict() for r in results],
            winner=winner,
            notes=notes
        )
    
    def run_all_comparisons(self, use_llm_judge: bool = True) -> list[ComparisonSummary]:
        """
        Run all test queries and return comparison summaries.
        """
        print("\n" + "="*60)
        print("RUNNING ALL BASELINE COMPARISONS")
        print("="*60)
        print(f"Test queries: {len(self.TEST_QUERIES)}")
        print(f"Hotel: {self.hotel_name} ({self.hotel_id})")
        print(f"City: {self.city}")
        
        summaries = []
        
        for i, test in enumerate(self.TEST_QUERIES, 1):
            print(f"\n\n{'#'*60}")
            print(f"TEST {i}/{len(self.TEST_QUERIES)}: {test['type']}")
            print('#'*60)
            
            summary = self.run_single_comparison(test["query"], use_llm_judge)
            summaries.append(summary)
        
        return summaries
    
    def generate_report(self, summaries: list[ComparisonSummary]) -> str:
        """
        Generate a comparison report from summaries.
        """
        report = []
        report.append("=" * 60)
        report.append("BASELINE COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Hotel: {self.hotel_name} ({self.hotel_id})")
        report.append(f"City: {self.city}")
        report.append(f"Total Tests: {len(summaries)}")
        report.append("")
        
        # Summary statistics
        winners = {"raw_data": 0, "simple_chatbot": 0, "full_system": 0}
        total_times = {"raw_data": [], "simple_chatbot": [], "full_system": []}
        
        for summary in summaries:
            if summary.winner:
                winners[summary.winner] = winners.get(summary.winner, 0) + 1
            
            for result in summary.results:
                if isinstance(result, dict):
                    total_times[result["system_type"]].append(result["elapsed_seconds"])
        
        report.append("--- OVERALL RESULTS ---")
        report.append("")
        report.append("Win Count:")
        for system, count in winners.items():
            report.append(f"  {system}: {count} wins")
        
        report.append("")
        report.append("Average Response Time:")
        for system, times in total_times.items():
            if times:
                avg = sum(times) / len(times)
                report.append(f"  {system}: {avg:.2f}s")
        
        report.append("")
        report.append("--- DETAILED RESULTS ---")
        
        for i, summary in enumerate(summaries, 1):
            report.append("")
            report.append(f"[Test {i}] Query: {summary.query}")
            report.append(f"  Winner: {summary.winner or 'N/A'}")
            report.append(f"  Notes: {summary.notes[:200]}..." if summary.notes else "  Notes: N/A")
            
            for result in summary.results:
                if isinstance(result, dict):
                    report.append(f"  - {result['system_type']}: {result['elapsed_seconds']:.2f}s")
                    if result.get('relevance_score'):
                        report.append(f"    Scores: R={result['relevance_score']}, A={result['actionability_score']}, Acc={result['accuracy_score']}")
        
        report.append("")
        report.append("=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, summaries: list[ComparisonSummary], filepath: str = None):
        """
        Save comparison results to JSON file.
        """
        if filepath is None:
            filepath = os.path.join(_PROJECT_ROOT, "baseline_comparison_results.json")
        
        data = {
            "metadata": {
                "hotel_id": self.hotel_id,
                "hotel_name": self.hotel_name,
                "city": self.city,
                "timestamp": datetime.now().isoformat()
            },
            "summaries": [asdict(s) for s in summaries]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
        return filepath


# ===========================================
# MAIN
# ===========================================

def run_baseline_comparison():
    """Run the baseline comparison tests."""
    
    # Default hotel context
    HOTEL_ID = "BKG_177691"
    HOTEL_NAME = "Malmaison London"
    CITY = "London"
    
    # Initialize comparison
    comparison = BaselineComparison(HOTEL_ID, HOTEL_NAME, CITY)
    
    # Run all tests
    summaries = comparison.run_all_comparisons(use_llm_judge=True)
    
    # Generate and print report
    report = comparison.generate_report(summaries)
    print("\n" + report)
    
    # Save results
    comparison.save_results(summaries)
    
    return summaries


if __name__ == "__main__":
    run_baseline_comparison()
