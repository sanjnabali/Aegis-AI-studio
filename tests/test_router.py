from backend.app.agents.router import agent_router

def test_router_code_intent():
    messages = [{"role": "user", "content": "Can you write a python script to parse JSON?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "code"

def test_router_search_intent():
    messages = [{"role": "user", "content": "What is the latest news about AI in 2024?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "web_search"

def test_router_chat_intent():
    messages = [{"role": "user", "content": "Hello, how are you today?"}]
    intent = agent_router.analyze_intent(messages)
    assert intent == "chat"