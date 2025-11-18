# test_app.py
from app import (
    UserRegistration,
    WeatherReporter,
    Notifier,
    PaymentProcessor
)

# ----------------------------
# 1. Dummy Example
# ----------------------------
def test_dummy(dummy_email_service):
    reg = UserRegistration(dummy_email_service)
    assert reg.register("alice") == "alice registered"


# ----------------------------
# 2. Stub Example
# ----------------------------
def test_stub(weather_stub):
    reporter = WeatherReporter(weather_stub)
    assert reporter.report("Paris") == "Paris: 20°C"


# ----------------------------
# 3. Fake Example
# ----------------------------
def test_fake(fake_user_repo):
    fake_user_repo.save("alice")
    fake_user_repo.save("bob")
    assert fake_user_repo.get_all() == ["alice", "bob"]


# ----------------------------
# 4. Spy Example
# ----------------------------
def test_spy(email_spy):
    notifier = Notifier(email_spy)
    notifier.notify("alice@example.com")

    assert email_spy.sent == [
        ("alice@example.com", "Hello", "Welcome!")
    ]


# ----------------------------
# 5. Mock Example
# ----------------------------
def test_mock(payment_gateway_mock):
    processor = PaymentProcessor(payment_gateway_mock)

    result = processor.charge(100)

    assert result == "ok"
    payment_gateway_mock.pay.assert_called_once_with(100)
