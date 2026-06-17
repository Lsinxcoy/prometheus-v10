from prometheus_v9pro.schema import Node, Genome, EvolutionDirection, HarnessDimension, HookPoint, HarnessConfig, HarnessEdit
from prometheus_v9pro.organs import OrganPipeline, TaotieOrgan
from prometheus_v9pro.engine import UnifiedEvolutionEngine
from prometheus_v9pro.events import EventBus
from prometheus_v9pro.store import SQLiteStore
print("ALL core imports OK")
print("HarnessDimension members:", [d.value for d in HarnessDimension])
print("HookPoint members:", [h.value for h in HookPoint])
hc = HarnessConfig()
print("HarnessConfig fingerprint:", hc.fingerprint())
he = HarnessEdit(action="replace", hook=HookPoint.BEFORE_MODEL, manifest={"changed_components": ["test"]})
print("HarnessEdit:", he.action, he.hook.value)
