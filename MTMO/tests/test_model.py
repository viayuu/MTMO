import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.MOEModel import MOEModel, MTL_Decoder
from models.MOELayer import MoE


def model_params(embedding_dim=16, head_num=2, qkv_dim=4):
    return dict(embedding_dim=embedding_dim, sqrt_embedding_dim=embedding_dim ** 0.5,
                encoder_layer_num=1, decoder_layer_num=1, head_num=head_num, qkv_dim=qkv_dim,
                ff_hidden_dim=32, num_experts=4, topk=2, eval_type='argmax', problem='Train_ALL',
                norm='instance', norm_loc='norm_last', expert_loc=['Enc0', 'Dec'],
                routing_level='node', routing_method='input_choice', device='cpu',
                logit_clipping=10, objective_scales=[8., 3.])


class ConditionalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_nonsquare_shapes_and_preference_cache(self):
        decoder = MTL_Decoder(**model_params())
        decoder.assign(torch.tensor([0.3, 0.7]))
        self.assertEqual(tuple(decoder.Wq_last_para.shape), (8, 22))
        self.assertEqual(tuple(decoder.Wk_para.shape), (8, 16))
        self.assertEqual(tuple(decoder.expert_weights.shape), (4, 16, 8))
        self.assertEqual(tuple(decoder.expert_biases.shape), (4, 16))
        decoder.set_kv(torch.randn(2, 6, 16))
        old_query = decoder.Wq_last_para.detach().clone()
        decoder.assign(torch.tensor([0.8, 0.2]))
        self.assertIsNone(decoder.k)
        self.assertIsNone(decoder.v)
        self.assertFalse(torch.equal(old_query, decoder.Wq_last_para))
        with self.assertRaises(RuntimeError):
            decoder(torch.randn(2, 3, 16), torch.randn(2, 3, 6), torch.zeros(2, 3, 6))
        full = MTL_Decoder(**model_params(128, 8, 16))
        full.assign(torch.tensor([0.5, 0.5]))
        self.assertEqual(tuple(full.Wq_last_para.shape), (128, 134))
        count = sum(x.numel() for x in (full.Wq_last_para, full.Wk_para, full.Wv_para,
                                       full.expert_weights, full.expert_biases))
        self.assertEqual(count, 115968)

    def test_complete_policy_has_generated_expert_and_encoder_gradients(self):
        batch, nodes, pomo = 8, 5, 4
        model = MOEModel(**model_params()).train()
        model.assign(torch.tensor([0.35, 0.65]))
        reset = SimpleNamespace(depot_xy=torch.rand(batch, 1, 2), node_xy=torch.rand(batch, nodes, 2),
                                node_demand=torch.rand(batch, nodes), node_tw_start=torch.zeros(batch, nodes),
                                node_tw_end=torch.ones(batch, nodes))
        model.pre_forward(reset)
        mask = torch.zeros(batch, pomo, nodes + 1)
        mask[:, :, -1] = float('-inf')
        state = SimpleNamespace(BATCH_IDX=torch.arange(batch)[:, None].expand(batch, pomo),
                                POMO_IDX=torch.arange(pomo)[None, :].expand(batch, pomo), selected_count=2,
                                current_node=torch.randint(nodes, (batch, pomo)), load=torch.rand(batch, pomo),
                                current_time=torch.rand(batch, pomo), length=torch.rand(batch, pomo),
                                open=torch.zeros(batch, pomo), total_distance=torch.rand(batch, pomo) * 8,
                                max_route_distance=torch.rand(batch, pomo) * 3, ninf_mask=mask)
        model.decoder.expert_biases.retain_grad()
        selected, prob = model(state, selected=torch.ones(batch, pomo, dtype=torch.long))
        self.assertEqual(tuple(selected.shape), (batch, pomo))
        self.assertTrue(torch.isfinite(prob).all())
        loss = -prob.log().mean() + model.aux_loss
        loss.backward()
        for name, parameter in model.decoder.named_parameters():
            if name.startswith('hyper_'):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(model.decoder.expert_biases.grad.abs().sum().item(), 0)
        self.assertGreater(model.decoder.multi_head_combine.w_gate.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.embedding_node.weight.grad.abs().sum().item(), 0)
        self.assertEqual(len(model.decoder.multi_head_combine.experts), 0)
        with torch.no_grad():
            probs, _ = model.decoder(torch.randn(batch, pomo, 16), torch.randn(batch, pomo, 6), mask)
        self.assertTrue((probs[:, :, -1] == 0).all())
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(batch, pomo))

    def test_generated_bias_and_original_dispatcher_agree(self):
        layer = MoE(6, 5, num_experts=4, k=2, noisy_gating=False,
                    moe_model='Linear', external_parameters=True).eval()
        x = torch.randn(2, 3, 6)
        weights = torch.zeros(4, 5, 6, requires_grad=True)
        biases = torch.arange(20, dtype=torch.float32).reshape(4, 5).requires_grad_()
        gates, _ = layer.noisy_top_k_gating(x.reshape(-1, 6), False)
        output, _ = layer(x, expert_weights=weights, expert_biases=biases)
        expected = (gates @ biases).reshape(2, 3, 5)
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertIsNotNone(weights.grad)
        self.assertGreater(biases.grad.abs().sum().item(), 0)
        # Ordinary MVMoE encoder/other-model experts retain their registered parameters.
        original = MoE(6, 5, num_experts=4, k=2, moe_model='Linear')
        original_output, aux = original(x)
        (original_output.square().mean() + aux).backward()
        self.assertEqual(len(original.experts), 4)
        self.assertTrue(any(p.grad is not None for p in original.experts.parameters()))

    def test_scale_buffer_and_preference_validation(self):
        model = MOEModel(**model_params())
        model.set_objective_scales([10., 2.])
        torch.testing.assert_close(model.state_dict()['objective_scales'], torch.tensor([10., 2.]))
        for scales in ([0., 1.], [1.], [float('nan'), 2.]):
            with self.assertRaises(ValueError):
                model.set_objective_scales(scales)
        for pref in ([0.1, 0.2], [-0.1, 1.1], [[0.5, 0.5]], [float('nan'), 1.]):
            with self.assertRaises(ValueError):
                model.assign(pref)

    def test_eval_repeat_and_checkpoint_scales(self):
        model = MOEModel(**model_params()).eval()
        reset = SimpleNamespace(depot_xy=torch.rand(2, 1, 2), node_xy=torch.rand(2, 5, 2),
                                node_demand=torch.rand(2, 5), node_tw_start=torch.zeros(2, 5),
                                node_tw_end=torch.ones(2, 5))
        last, attr, mask = torch.rand(2, 3, 16), torch.rand(2, 3, 6), torch.zeros(2, 3, 6)
        with torch.no_grad():
            model.assign([0.4, 0.6])
            model.pre_forward(reset)
            expected, _ = model.decoder(last, attr, mask)
            model.assign([0.4, 0.6])
            model.pre_forward(reset)
            repeated, _ = model.decoder(last, attr, mask)
        torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
        checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
        params = model_params()
        params['objective_scales'] = [1., 1.]
        restored = MOEModel(**params).eval()
        restored.load_state_dict(checkpoint, strict=True)
        torch.testing.assert_close(restored.objective_scales, torch.tensor([8., 3.]))
        with torch.no_grad():
            restored.assign([0.4, 0.6])
            restored.pre_forward(reset)
            actual, _ = restored.decoder(last, attr, mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
