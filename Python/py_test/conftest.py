# conftest.py
import pytest
from unittest.mock import Mock
from app import (
    WeatherService, EmailSender, PaymentGateway,
    InMemoryUserRepository
)

# -------------------------------
# Dummy Fixture
# -------------------------------
class DummyEmailService:
    pass

@pytest.fixture
def dummy_email_service():
    return DummyEmailService()


# -------------------------------
# Stub Fixture
# -------------------------------
class WeatherServiceStub(WeatherService):
    def get_temperature(self, city):
        return 20

@pytest.fixture
def weather_stub():
    return WeatherServiceStub()


# -------------------------------
# Fake Fixture
# -------------------------------
@pytest.fixture
def fake_user_repo():
    # Fake from app.py
    return InMemoryUserRepository()


# -------------------------------
# Spy Fixture
# -------------------------------
class EmailSenderSpy(EmailSender):
    def __init__(self):
        self.sent = []

    def send(self, to, subject, body):
        self.sent.append((to, subject, body))

@pytest.fixture
def email_spy():
    return EmailSenderSpy()


# -------------------------------
# Mock Fixture
# -------------------------------
@pytest.fixture
def payment_gateway_mock():
    mock = Mock(spec=PaymentGateway)
    mock.pay.return_value = "ok"
    return mock
