"""Check SafetyManager and InitiativeLayer return formats."""
from prometheus_v10.manager import SafetyManager
from prometheus_v10.initiative import InitiativeLayer
from prometheus_v10.autonomy import AutonomyManager
from prometheus_v10.trust import TrustManager

# SafetyManager
sm = SafetyManager()
r1 = sm.check("search", "def safe_function(): return 42")
print("Safe result:", r1)
print("Safe keys:", list(r1.keys()) if isinstance(r1, dict) else type(r1))

r2 = sm.check("exec", "import os; os.system('rm -rf /')")
print("Dangerous result:", r2)
print("Dangerous keys:", list(r2.keys()) if isinstance(r2, dict) else type(r2))

# InitiativeLayer
am = AutonomyManager()
tm = TrustManager()
il = InitiativeLayer(am, tm, sm)
r3 = il.propose("read", "safe content")
print("Initiative result:", r3)
print("Initiative type:", type(r3))
if isinstance(r3, dict):
    print("Initiative keys:", list(r3.keys()))
