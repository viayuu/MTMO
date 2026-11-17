import torch
import torch.nn as nn
import torch.nn.functional as F
# from tutel import moe as tutel_moe
from .MOELayer import MoE

__all__ = ['MOEModel']


class MOEModel(nn.Module):
    """
        MOE implementations:
            (1) with tutel, ref to "https://github.com/microsoft/tutel"
            (2) with "https://github.com/davidmrau/mixture-of-experts"
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']
        self.aux_loss = 0

        self.encoder = MTL_Encoder(**model_params)
        self.decoder = MTL_Decoder(**model_params)
        self.encoded_nodes = None  # shape: (batch, problem+1, EMBEDDING_DIM)
        device = model_params.get('device')
        if device is None:
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        scales = torch.as_tensor(model_params.get('objective_scales', [1., 1.]), dtype=torch.float32)
        if scales.shape != (2,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("objective_scales must contain two finite positive values")
        self.register_buffer('objective_scales', scales.clone())

    def set_objective_scales(self, scales):
        scales = torch.as_tensor(scales, device=self.objective_scales.device, dtype=self.objective_scales.dtype)
        if scales.shape != (2,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("objective_scales must contain two finite positive values")
        self.objective_scales.copy_(scales)

    def assign(self, pref):
        self.decoder.assign(pref)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        # shape: (batch, problem)
        node_xy_demand_tw = torch.cat((node_xy, node_demand[:, :, None], node_tw_start[:, :, None], node_tw_end[:, :, None]), dim=2)
        # shape: (batch, problem, 5)
        # prob_emb = reset_state.prob_emb
        # shape: (1, 5) - only for problem-level routing/gating

        self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
        self.aux_loss = moe_loss
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=self.encoded_nodes.device)
            prob = torch.ones(size=(batch_size, pomo_size), device=self.encoded_nodes.device)
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))
            # shape: (batch, pomo, problem_size+1)

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            # selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, -1).to(self.device)
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size), device=self.encoded_nodes.device)
            # probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            attr = torch.cat((state.load[:, :, None], state.current_time[:, :, None], state.length[:, :, None], state.open[:, :, None],
                              (state.total_distance / self.objective_scales[0])[:, :, None],
                              (state.max_route_distance / self.objective_scales[1])[:, :, None]), dim=2)
            # shape: (batch, pomo, 6)
            probs, moe_loss = self.decoder(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            self.aux_loss += moe_loss
            # shape: (batch, pomo, problem+1)
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            raise RuntimeError("Invalid action probabilities for {}".format(state.PROBLEM)) from exception
                    else:
                        selected = probs.argmax(dim=2)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
            else:
                selected = selected
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        hidden_dim = self.model_params['ff_hidden_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # [Option 1]: Use MoEs in Raw Features
        if self.model_params['num_experts'] > 1 and "Raw" in self.model_params['expert_loc']:
            self.embedding_depot = MoE(input_size=2, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                       k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                       routing_method=self.model_params['routing_method'], moe_model="Linear")
            self.embedding_node = MoE(input_size=5, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                      k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                      routing_method=self.model_params['routing_method'], moe_model="Linear")
        else:
            self.embedding_depot = nn.Linear(2, embedding_dim)
            self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(i, **model_params) for i in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_tw):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_tw.shape: (batch, problem, 5)
        # prob_emb: (1, embedding)

        moe_loss = 0
        if isinstance(self.embedding_depot, MoE) or isinstance(self.embedding_node, MoE):
            embedded_depot, loss_depot = self.embedding_depot(depot_xy)
            embedded_node, loss_node = self.embedding_node(node_xy_demand_tw)
            moe_loss = moe_loss + loss_depot + loss_node
        else:
            embedded_depot = self.embedding_depot(depot_xy)
            # shape: (batch, 1, embedding)
            embedded_node = self.embedding_node(node_xy_demand_tw)
            # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out, loss = layer(out)
            moe_loss = moe_loss + loss

        return out, moe_loss
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, depth=0, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        # [Option 2]: Use MoEs in Encoder
        if self.model_params['num_experts'] > 1 and "Enc{}".format(depth) in self.model_params['expert_loc']:
            # TODO: enabling parallelism
            # (1) MOE with tutel, ref to "https://github.com/microsoft/tutel"
            """
            assert self.model_params['routing_level'] == "node", "Tutel only supports node-level routing!"
            self.feedForward = tutel_moe.moe_layer(
                gate_type={'type': 'top', 'k': self.model_params['topk']},
                model_dim=embedding_dim,
                experts={'type': 'ffn', 'count_per_node': self.model_params['num_experts'],
                         'hidden_size_per_expert': self.model_params['ff_hidden_dim'],
                         'activation_fn': lambda x: F.relu(x)},
            )
            """
            # (2) MOE with "https://github.com/davidmrau/mixture-of-experts"
            self.feedForward = MoE(input_size=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
        else:
            self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        """
        Two implementations:
            norm_last: the original implementation of AM/POMO: MHA -> Add & Norm -> FFN/MOE -> Add & Norm
            norm_first: the convention in NLP: Norm -> MHA -> Add -> Norm -> FFN/MOE -> Add
        """
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num, moe_loss = self.model_params['head_num'], 0

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)  # (batch, problem, HEAD_NUM*KEY_DIM)
            multi_head_out = self.multi_head_combine(out_concat)  # (batch, problem, EMBEDDING_DIM)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2, moe_loss = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)  # (batch, problem, EMBEDDING_DIM)
        else:
            out1 = self.addAndNormalization1(None, input1)
            multi_head_out = self.multi_head_combine(out1)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2, moe_loss = self.feedForward(out2)
            out3 = input2 + out2

        return out3, moe_loss


########################################
# DECODER
########################################

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        projection_dim = head_num * qkv_dim
        self.num_output_experts = self.model_params['num_experts'] if (
            self.model_params['num_experts'] > 1 and 'Dec' in self.model_params['expert_loc']) else 1

        # PMOCO parameter compression, with the paper's nonlinear 128-wide trunk.
        self.hyper_fc1 = nn.Linear(2, 128)
        self.hyper_fc2 = nn.Linear(128, 128)
        self.hyper_fc3 = nn.Linear(128, 2 * (3 + self.num_output_experts))
        self.hyper_Wq_last = nn.Linear(2, projection_dim * (embedding_dim + 6), bias=False)
        self.hyper_Wk = nn.Linear(2, projection_dim * embedding_dim, bias=False)
        self.hyper_Wv = nn.Linear(2, projection_dim * embedding_dim, bias=False)
        self.hyper_multi_head_combine = nn.ModuleList([
            nn.Linear(2, embedding_dim * projection_dim + embedding_dim, bias=False)
            for _ in range(self.num_output_experts)])

        # Preserve MVMoE routing, using the generated affine experts.
        if self.num_output_experts > 1:
            self.multi_head_combine = MoE(input_size=projection_dim, output_size=embedding_dim,
                                          num_experts=self.num_output_experts,
                                          k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                          routing_level=self.model_params['routing_level'],
                                          routing_method=self.model_params['routing_method'],
                                          moe_model="Linear", external_parameters=True)
        else:
            self.multi_head_combine = None

        self.Wq_last_para = None
        self.Wk_para = None
        self.Wv_para = None
        self.expert_weights = None
        self.expert_biases = None
        self.k = None
        self.v = None
        self.single_head_key = None

    def assign(self, pref):
        pref = torch.as_tensor(pref, device=self.hyper_fc1.weight.device, dtype=self.hyper_fc1.weight.dtype)
        if pref.shape != (2,) or not torch.isfinite(pref).all() or (pref < 0).any():
            raise ValueError("A batch shares one finite nonnegative two-objective preference")
        if not torch.isclose(pref.sum(), pref.new_tensor(1.)):
            raise ValueError("Preference values must sum to one")
        latent = self.hyper_fc3(F.relu(self.hyper_fc2(F.relu(self.hyper_fc1(pref)))))
        groups = latent.reshape(3 + self.num_output_experts, 2)
        embedding_dim = self.model_params['embedding_dim']
        projection_dim = self.model_params['head_num'] * self.model_params['qkv_dim']
        # F.linear weights always have [out_features, in_features] shape.
        self.Wq_last_para = self.hyper_Wq_last(groups[0]).reshape(projection_dim, embedding_dim + 6)
        self.Wk_para = self.hyper_Wk(groups[1]).reshape(projection_dim, embedding_dim)
        self.Wv_para = self.hyper_Wv(groups[2]).reshape(projection_dim, embedding_dim)
        expert_params = [head(groups[3 + i]) for i, head in enumerate(self.hyper_multi_head_combine)]
        self.expert_weights = torch.stack([
            x[:embedding_dim * projection_dim].reshape(embedding_dim, projection_dim) for x in expert_params])
        self.expert_biases = torch.stack([x[embedding_dim * projection_dim:] for x in expert_params])
        # A new preference invalidates cached keys and values from the preceding one.
        self.k = self.v = self.single_head_key = None

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        if self.Wq_last_para is None:
            raise RuntimeError("Call model.assign(pref) before pre_forward/set_kv")
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, attr, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # attr.shape: (batch, pomo, 6)
        # ninf_mask.shape: (batch, pomo, problem+1)
        if self.k is None or self.v is None:
            raise RuntimeError("Call pre_forward/set_kv after assigning the current preference")
        head_num, moe_loss = self.model_params['head_num'], 0

        # Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q = reshape_by_heads(F.linear(input_cat, self.Wq_last_para), head_num=head_num)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)

        if isinstance(self.multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.multi_head_combine(
                out_concat, expert_weights=self.expert_weights, expert_biases=self.expert_biases)
        else:
            mh_atten_out = F.linear(out_concat, self.expert_weights[0], self.expert_biases[0])

        # Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / self.model_params['sqrt_embedding_dim']
        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        return probs, moe_loss


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        # input.shape: (batch, problem, embedding)
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            # shape: (batch, embedding, problem)
            normalized = self.norm(transposed)
            # shape: (batch, embedding, problem)
            back_trans = normalized.transpose(1, 2)
            # shape: (batch, problem, embedding)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1))), 0
