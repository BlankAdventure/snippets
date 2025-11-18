# test_app.py
from unittest.mock import Mock
from app import (
    UserRegistration,
    WeatherService, WeatherReporter,    
    InMemoryUserRepository,
    EmailSender, Notifier,
    PaymentGateway, PaymentProcessor
)

# ----------------------------
# 1. Dummy Example Test
# ----------------------------


class DummyEmailService:
    """Never used — only satisfies __init__ requirement."""
    pass


def test_dummy_example():
    dummy = DummyEmailService()
    reg = UserRegistration(dummy)

    assert reg.register("alice") == "alice registered"


# ----------------------------
# 2. Stub Example Test
# ----------------------------

class WeatherServiceStub(WeatherService):
    def get_temperature(self, city):
        return 20  # fixed output


def test_stub_example():
    stub = WeatherServiceStub()
    reporter = WeatherReporter(stub)

    assert reporter.report("Paris") == "Paris: 20°C"


# ----------------------------
# 3. Fake Example Test
# ----------------------------

def test_fake_example():
    repo = InMemoryUserRepository()
    repo.save("alice")
    repo.save("bob")

    assert repo.get_all() == ["alice", "bob"]


# ----------------------------
# 4. Spy Example Test
# ----------------------------

class EmailSenderSpy(EmailSender):
    def __init__(self):
        self.sent = []

    def send(self, to, subject, body):
        self.sent.append((to, subject, body))


def test_spy_example():
    spy = EmailSenderSpy()
    notifier = Notifier(spy)

    notifier.notify("alice@example.com")

    assert spy.sent == [
        ("alice@example.com", "Hello", "Welcome!")
    ]


# ----------------------------
# 5. Mock Example Test
# ----------------------------

def test_mock_example():
    gateway_mock = Mock(spec=PaymentGateway)
    gateway_mock.pay.return_value = "ok"

    processor = PaymentProcessor(gateway_mock)

    result = processor.charge(100)

    assert result == "ok"
    gateway_mock.pay.assert_called_once_with(100)
