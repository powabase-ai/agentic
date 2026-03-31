from agentic.agent.rules import evaluate_rules


class TestEvaluateRules:
    def test_contains_deny(self):
        rules = [
            {
                "condition": "query CONTAINS 'DROP'",
                "action": "deny",
                "message": "No DROP",
            }
        ]
        allowed, reason = evaluate_rules({"query": "DROP TABLE users"}, rules)
        assert allowed is False
        assert reason == "No DROP"

    def test_contains_allow(self):
        rules = [
            {
                "condition": "query CONTAINS 'DROP'",
                "action": "deny",
                "message": "No DROP",
            }
        ]
        allowed, reason = evaluate_rules({"query": "SELECT * FROM users"}, rules)
        assert allowed is True
        assert reason is None

    def test_not_contains(self):
        rules = [
            {
                "condition": "url NOT CONTAINS 'internal'",
                "action": "deny",
                "message": "Must be internal",
            }
        ]
        # URL without 'internal' → condition matches → deny
        allowed, _ = evaluate_rules({"url": "https://public.example.com"}, rules)
        assert allowed is False
        # URL with 'internal' → condition doesn't match → allow
        allowed, _ = evaluate_rules({"url": "https://internal.example.com"}, rules)
        assert allowed is True

    def test_starts_with(self):
        rules = [
            {
                "condition": "query STARTS_WITH 'SELECT'",
                "action": "deny",
                "message": "Only SELECT",
            }
        ]
        # This rule denies if it DOES start with SELECT — but that's backwards
        # Let's fix: the rule should deny if it does NOT start with SELECT
        rules = [
            {
                "condition": "query NOT STARTS_WITH 'SELECT'",
                "action": "deny",
                "message": "Only SELECT allowed",
            }
        ]
        allowed, _ = evaluate_rules({"query": "SELECT 1"}, rules)
        assert allowed is True
        allowed, reason = evaluate_rules({"query": "DELETE FROM users"}, rules)
        assert allowed is False

    def test_in_list(self):
        rules = [
            {
                "condition": "method IN ['GET', 'HEAD']",
                "action": "deny",
                "message": "Read only",
            }
        ]
        allowed, _ = evaluate_rules({"method": "GET"}, rules)
        assert allowed is False
        allowed, _ = evaluate_rules({"method": "POST"}, rules)
        assert allowed is True

    def test_not_in_list(self):
        rules = [
            {
                "condition": "method NOT IN ['GET', 'HEAD']",
                "action": "deny",
                "message": "Must be read-only",
            }
        ]
        allowed, _ = evaluate_rules({"method": "POST"}, rules)
        assert allowed is False
        allowed, _ = evaluate_rules({"method": "GET"}, rules)
        assert allowed is True

    def test_matches_regex(self):
        rules = [
            {
                "condition": "query MATCHES '.*DROP.*'",
                "action": "deny",
                "message": "No DROP",
            }
        ]
        allowed, _ = evaluate_rules({"query": "SELECT 1; DROP TABLE x"}, rules)
        assert allowed is False

    def test_missing_field_passes(self):
        rules = [
            {
                "condition": "query CONTAINS 'DROP'",
                "action": "deny",
                "message": "No DROP",
            }
        ]
        allowed, _ = evaluate_rules({"other_field": "value"}, rules)
        assert allowed is True

    def test_empty_rules(self):
        allowed, reason = evaluate_rules({"query": "anything"}, [])
        assert allowed is True

    def test_first_deny_wins(self):
        rules = [
            {
                "condition": "query CONTAINS 'SELECT'",
                "action": "deny",
                "message": "First",
            },
            {
                "condition": "query CONTAINS 'DROP'",
                "action": "deny",
                "message": "Second",
            },
        ]
        allowed, reason = evaluate_rules({"query": "SELECT 1"}, rules)
        assert reason == "First"
