# app.py

# ---------- Dummy Example ----------
class EmailService:
    def send(self, to, subject, body):
        raise NotImplementedError()


class UserRegistration:
    def __init__(self, email_service):
        self.email_service = email_service

    def register(self, username):
        # Does not call email_service here
        return f"{username} registered"


# ---------- Stub Example ----------
class WeatherService:
    def get_temperature(self, city):
        raise NotImplementedError()


class WeatherReporter:
    def __init__(self, service):
        self.service = service

    def report(self, city):
        temp = self.service.get_temperature(city)
        return f"{city}: {temp}°C"


# ---------- Fake Example ----------
class UserRepository:
    def save(self, username):
        raise NotImplementedError()

    def get_all(self):
        raise NotImplementedError()


# A fake implementation of the repository
class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.users = []

    def save(self, username):
        self.users.append(username)

    def get_all(self):
        return list(self.users)


# ---------- Spy Example ----------
class EmailSender:
    def send(self, to, subject, body):
        raise NotImplementedError()


class Notifier:
    def __init__(self, email_sender):
        self.email_sender = email_sender

    def notify(self, user):
        self.email_sender.send(user, "Hello", "Welcome!")


# ---------- Mock Example ----------
class PaymentGateway:
    def pay(self, amount):
        raise NotImplementedError()


class PaymentProcessor:
    def __init__(self, gateway):
        self.gateway = gateway

    def charge(self, amount):
        return self.gateway.pay(amount)
