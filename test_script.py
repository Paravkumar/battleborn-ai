import sys
sys.path.insert(0, 'src')

from ticket_agent.config import Settings
from ticket_agent.repository import TicketRepository
from ticket_agent.knowledge_base import KnowledgeBase
from ticket_agent.tools import TicketResolutionToolkit

# Initialize components
settings = Settings()
repository = TicketRepository(settings.data_dir)
kb = KnowledgeBase(settings.data_dir)

toolkit = TicketResolutionToolkit(
    repository=repository,
    kb=kb,
    llm=None,
    student_model=settings.student_model,
    use_ollama=False,
    min_kb_confidence=settings.min_kb_confidence,
    min_classification_confidence=settings.min_classification_confidence,
)

# Test full workflow for TICK-1004
ticket_id = "TICK-1004"

# Step 1: Classify
classify = toolkit.classify_ticket(ticket_id)
print(f"1. Classify: {classify.data}")

# Step 2: Search KB
search = toolkit.search_kb(
    ticket_id=ticket_id,
    query="Unable to log into account i left my account 2 years ago now when i am trying to log in i cant",
    issue_type=classify.data.get('issue_type'),
    intent=classify.data.get('intent'),
    classification_confidence=classify.data.get('confidence_score'),
    limit=3
)
print(f"\n2. KB Search success: {search.success}")
if search.data and 'hits' in search.data:
    for hit in search.data['hits']:
        print(f"   - {hit['article_id']}: {hit['title']}")

# Step 3: Draft response
if search.success and search.data.get('hits'):
    article_ids = [h['article_id'] for h in search.data['hits']]
    draft = toolkit.draft_response(ticket_id=ticket_id, article_ids=article_ids)
    print(f"\n3. Draft success: {draft.success}")
    print(f"   Message: {draft.message}")
    if draft.data:
        print(f"   Body: {draft.data.get('body', 'N/A')[:200]}...")
        print(f"   Citations: {draft.data.get('citations', [])}")
        print(f"   Needs escalation: {draft.data.get('needs_escalation', 'N/A')}")
