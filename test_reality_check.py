# test_reality_check.py
print("🔍 REALITY CHECK: Testing actual component instantiation")
print("=" * 60)

# Test 1: Gaussian Process
print("\n1. Testing Gaussian Process:")
try:
    from src.models.gaussian_process import GaussianProcess
    print("✅ GP import successful")
    gp = GaussianProcess()
    print("✅ GP instantiation successful")
    print(f"   GP type: {type(gp)}")
    print(f"   GP methods: {[m for m in dir(gp) if not m.startswith('_')]}")
except Exception as e:
    print(f"❌ GP failed: {e}")

# Test 2: MPC Controller  
print("\n2. Testing MPC Controller:")
try:
    from src.controllers.mpc_controller import MPCController
    print("✅ MPC import successful")
    # Try to instantiate - this should fail because it's abstract
    mpc = MPCController()
    print("✅ MPC instantiation successful")
    print(f"   MPC type: {type(mpc)}")
except Exception as e:
    print(f"❌ MPC failed: {e}")

# Test 3: Human Behavior Model
print("\n3. Testing Human Behavior Model:")
try:
    from src.models.human_behavior import HumanBehaviorModel
    print("✅ Human Behavior import successful")
    # Try to instantiate - this should fail because it's abstract
    hb = HumanBehaviorModel({})
    print("✅ Human Behavior instantiation successful")
    print(f"   HB type: {type(hb)}")
except Exception as e:
    print(f"❌ Human Behavior failed: {e}")

# Test 4: Bayesian RL Agent
print("\n4. Testing Bayesian RL Agent:")
try:
    from src.agents.bayesian_rl_agent import BayesianRLAgent
    print("✅ Bayesian RL import successful")
    agent = BayesianRLAgent()
    print("✅ Bayesian RL instantiation successful")  
    print(f"   Agent type: {type(agent)}")
except Exception as e:
    print(f"❌ Bayesian RL failed: {e}")

# Test 5: System Integration
print("\n5. Testing System Integration:")
try:
    from src.system.human_intent_rl_system import HumanIntentRLSystem
    print("✅ System import successful")
    system = HumanIntentRLSystem()
    print("✅ System instantiation successful")
    print(f"   System type: {type(system)}")
except Exception as e:
    print(f"❌ System failed: {e}")

print("\n" + "=" * 60)
print("🎯 REALITY CHECK COMPLETE")