#!/usr/bin/env python3
"""
Fix the training_step method in LightningDDPOModule.
The issue: training_step returns None, which breaks Lightning's training loop.
Solution: Return the loss tensor instead.
"""

import sys

# Read the file
with open('/home/md666/data/disk6/DiffB110/ddpo/trainer/lightning_ddpo_module.py', 'r') as f:
    content = f.read()

# Replace the problematic return statement
old_code = """        # Return None to tell Lightning we handled optimization ourselves
        return None"""

new_code = """        # Return loss tensor for Lightning to continue training
        return loss"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Fixed: Changed 'return None' to 'return loss'")
else:
    print("❌ Could not find the old code pattern")
    sys.exit(1)

# Also fix configure_optimizers to enable automatic optimization
old_optimizer = """    def configure_optimizers(self):
        """
        Configure optimizer for PPO training.
        
        IMPORTANT: EDMActor already has its own optimizer that it manages internally.
        We return None to tell Lightning to use manual optimization mode.
        
        Returns:
            None: Manual optimization (actor handles optimizer)
        """
        # Tell Lightning we're using manual optimization
        self.automatic_optimization = False
        return None"""

new_optimizer = """    def configure_optimizers(self):
        """
        Configure optimizer for PPO training.
        
        IMPORTANT: EDMActor already has its own optimizer that it manages internally.
        We return None to tell Lightning to use manual optimization mode.
        
        Returns:
            None: Manual optimization (actor handles optimizer)
        """
        # Tell Lightning we're using manual optimization
        self.automatic_optimization = False
        # Return a dummy optimizer so Lightning doesn't complain
        return torch.optim.SGD(self.model.parameters(), lr=0.001)"""

if old_optimizer in content:
    content = content.replace(old_optimizer, new_optimizer)
    print("✅ Fixed: Updated configure_optimizers to return a dummy optimizer")
else:
    print("⚠️  Could not find configure_optimizers pattern (may already be fixed)")

# Write back
with open('/home/md666/data/disk6/DiffB110/ddpo/trainer/lightning_ddpo_module.py', 'w') as f:
    f.write(content)

print("✅ File updated successfully!")
