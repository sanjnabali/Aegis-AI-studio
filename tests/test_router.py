from app.agents.router import agent_router

def test_router_code_intent():
    """Test that code-related queries are routed to code agent"""
    messages = [{"role": "user", "content": "Can you write a python script to parse JSON?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "code"

def test_router_search_intent():
    """Test that current event queries are routed to web search agent"""
    messages = [{"role": "user", "content": "What is the latest news about AI in 2024?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "web_search"

def test_router_chat_intent():
    """Test that general chat queries are routed to chat agent"""
    # Removed "today" and other web_search triggers to ensure chat routing
    messages = [{"role": "user", "content": "Hello, how are you doing?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "chat"

def test_router_web_search_with_time_keywords():
    """Test that queries with time keywords trigger web search"""
    messages = [{"role": "user", "content": "What happened this week in tech?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "web_search"