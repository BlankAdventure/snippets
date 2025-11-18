# test_app_param.py
import pytest
from app import (    
    WeatherReporter,
    PaymentProcessor
)


# =====================================================
# 1. Parameterized Stub Example
# =====================================================
# Uses weather_stub, but we override the stub inside the test
# through a parameterized inner Stub class.
@pytest.mark.parametrize("city,temp,expected", [
    ("Paris", 20, "Paris: 20°C"),
    ("Berlin", 15, "Berlin: 15°C"),
    ("Tokyo", 30, "Tokyo: 30°C"),
])
def test_stub_parameterized(city, temp, expected):
    class ParamStub:
        def get_temperature(self, c):
            assert c == city
            return temp
    
    reporter = WeatherReporter(ParamStub())
    assert reporter.report(city) == expected


# =====================================================
# 2. Parameterized Fake Example
# =====================================================
@pytest.mark.parametrize("users", [
    ["alice"],
    ["bob", "charlie"],
    ["dave", "eve", "frank"],
])
def test_fake_parameterized(fake_user_repo, users):
    for u in users:
        fake_user_repo.save(u)

    assert fake_user_repo.get_all() == users


# =====================================================
# 3. Parameterized Mock Example
# =====================================================
@pytest.mark.parametrize("amount,return_value", [
    (10, "ok"),
    (999, "high"),
    (0, "free"),
])
def test_mock_parameterized(payment_gateway_mock, amount, return_value):
    payment_gateway_mock.pay.return_value = return_value

    processor = PaymentProcessor(payment_gateway_mock)
    result = processor.charge(amount)

    assert result == return_value
    payment_gateway_mock.pay.assert_called_once_with(amount)
