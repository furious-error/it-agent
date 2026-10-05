from app.compaction import COMPACTION_THRESHOLD, compact_messages
from app.messages import content_from_dict, content_to_dict, user_text


def _user(text: str) -> dict:
    return user_text(text)


def _call(name: str) -> dict:
    return {"role": "model", "parts": [{"function_call": {"name": name, "args": {"service": "orders-api"}, "id": "c"}}]}


def _result(name: str) -> dict:
    return {"role": "user", "parts": [{"function_response": {"name": name, "id": "c", "response": {"status": 200, "result": {}}}}]}


def test_threshold_is_ten_messages():
    assert COMPACTION_THRESHOLD == 10
    seen = []

    def summarize(transcript: str) -> str:
        seen.append(transcript)
        return "should not run"

    summary, messages = compact_messages([_user(str(i)) for i in range(10)], summarize)
    assert summary is None and len(messages) == 10 and seen == []


def test_the_oldest_messages_become_one_summary():
    seen = []

    def summarize(transcript: str) -> str:
        seen.append(transcript)
        return "orders-api degraded after deploy-8472"

    original = [_user(f"note {i}") for i in range(12)]
    summary, messages = compact_messages(original, summarize)
    assert summary == "orders-api degraded after deploy-8472"
    assert messages[0]["parts"][0]["text"].startswith("Summary of the investigation so far:")
    assert "deploy-8472" in messages[0]["parts"][0]["text"]
    assert messages[1:] == original[8:]
    assert len(messages) == 5
    assert "note 0" in seen[0] and "note 7" in seen[0] and "note 8" not in seen[0]


def test_a_function_response_is_not_split_from_its_call():
    # Eleven messages, and a naive cut at index 8 would start on a function response.
    messages = []
    for i in range(4):
        messages.append(_user(f"q{i}"))
        messages.append(_call(f"tool{i}"))
    messages.append(_result("tool3"))
    messages.append(_user("tail-a"))
    messages.append(_user("tail-b"))
    summary, compacted = compact_messages(messages, lambda transcript: "rolled up")
    assert summary == "rolled up"
    assert compacted[0]["parts"][0]["text"].startswith("Summary")
    assert compacted[-2:] == [_user("tail-a"), _user("tail-b")]
    assert "function_response" not in str(compacted)


def test_existing_summary_is_included_in_the_next_compaction():
    captured = []

    def summarize(transcript: str) -> str:
        captured.append(transcript)
        return "combined"

    compact_messages([_user(str(i)) for i in range(11)], summarize, existing_summary="earlier finding")
    assert captured[0].startswith("Previous summary:\nearlier finding")


def test_message_round_trip_keeps_the_function_call_and_signature():
    from google.genai import types

    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_service_health", args={"service": "orders-api"}, id="c1"),
                thought_signature=b"keep-me",
            )
        ],
    )
    restored = content_from_dict(content_to_dict(content))
    assert restored.parts[0].function_call.name == "get_service_health"
    assert restored.parts[0].function_call.id == "c1"
    assert restored.parts[0].thought_signature == b"keep-me"
    assert content_from_dict(user_text("hello")).parts[0].text == "hello"
