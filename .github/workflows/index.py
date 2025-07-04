import json
import numpy as np

# ---- Begin IAI_IPS_Prototype_V2 definition ----
def rbf_kernel(dist, gamma):
    return np.exp(-gamma * (dist**2))

class IAI_IPS_Prototype_V2:
    def __init__(self, learning_rate_cn=0.1):
        # Refined archetypes
        self.V_raw = np.array([0.1, 0.1])
        self.V_ripe = np.array([0.9, 0.9])
        self.gamma = 2.0
        # Random weights for layers
        self.W4 = np.random.randn(5, 5) * 0.1
        self.W5 = np.random.randn(5, 5) * 0.1
        self.W6 = np.random.randn(5, 5) * 0.1
        self.W7 = np.random.randn(5, 4) * 0.1
        self.W8 = np.random.randn(4, 3) * 0.1
        self.W9 = np.random.randn(3, 2) * 0.1
        self.W10 = np.random.randn(2, 1) * 0.1
        self.core_node_alpha = 0.1
        self.learning_rate_cn = learning_rate_cn

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, features):
        l1_output = features
        dist_to_raw = np.linalg.norm(l1_output - self.V_raw)
        dist_to_ripe = np.linalg.norm(l1_output - self.V_ripe)
        a_raw = rbf_kernel(dist_to_raw, self.gamma)
        a_ripe = rbf_kernel(dist_to_ripe, self.gamma)
        l2_output = np.array([a_raw, a_ripe])

        a_int_contradiction = 1 - abs(a_raw - a_ripe)
        a_ext_contradiction = 1 - a_ripe
        a_intext_contradiction = a_int_contradiction * a_ext_contradiction
        l3_output = np.array([a_int_contradiction, a_ext_contradiction, a_intext_contradiction])

        delta_alpha = self.learning_rate_cn * a_intext_contradiction
        self.core_node_alpha = np.clip(self.core_node_alpha + delta_alpha, 0.0, 1.0)

        l_2_3_combined = np.concatenate((l2_output, l3_output))
        l4_output = self.sigmoid(np.dot(l_2_3_combined, self.W4.T))
        l5_output = self.sigmoid(np.dot(l4_output, self.W5.T))
        cn_scaling_factors = 1 + (self.core_node_alpha * l5_output)
        adjusted_l5_output = l5_output * cn_scaling_factors

        l6_output = self.sigmoid(np.dot(adjusted_l5_output, self.W6.T))
        l7_output = self.sigmoid(np.dot(l6_output, self.W7.T))
        l8_output = self.sigmoid(np.dot(l7_output, self.W8.T))
        l9_output = self.sigmoid(np.dot(l8_output, self.W9.T))
        final_output = self.sigmoid(np.dot(l9_output, self.W10.T))

        return final_output[0], {
            "L2_Raw": a_raw,
            "L2_Ripe": a_ripe,
            "L3_IntCont": a_int_contradiction,
            "CoreNode_Alpha": self.core_node_alpha,
        }
# ---- End IAI_IPS_Prototype_V2 definition ----

iai_ips_v2 = IAI_IPS_Prototype_V2()

def handler(request, context):
    try:
        body = request.get_json()
        fruit = np.array(body.get("fruit", [0.1, 0.2]))
        output, details = iai_ips_v2.forward(fruit)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "input": body.get("fruit"),
                "output": float(output),
                "details": {k: float(v) for k, v in details.items()}
            }),
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
        }