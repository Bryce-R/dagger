def rollout_with_policy(policy_fn, x0, horizon=50):
    xs = [x0]
    us = []
    x = x0
    for t in range(horizon):
        u = policy_fn(x, xs)
        us.append(u)
        noise = rng.normal(0.0, PROCESS_NOISE_STD)
        x = x + u * DT + noise
        xs.append(x)
    return np.array(xs), np.array(us)


def expert_policy_fn(x, xs_history):
    return expert_policy(x)


def learned_policy_fn_factory(model, context_len=CONTEXT_LEN):
    def policy(x, xs_history):
        # xs_history includes current x at the end
        # Take last context_len states, pad if necessary
        x_hist = np.array(xs_history, dtype=np.float32)
        if len(x_hist) < context_len:
            pad = np.zeros(context_len - len(x_hist), dtype=np.float32)
            x_in = np.concatenate([pad, x_hist])
        else:
            x_in = x_hist[-context_len:]
        x_tensor = torch.from_numpy(x_in).view(1, context_len, 1).to(device)
        with torch.no_grad():
            u_pred = model(x_tensor).cpu().item()
        # Optionally clip to same range as expert
        u_pred = float(np.clip(u_pred, -U_MAX, U_MAX))
        return u_pred
    return policy


# ---------- Compare rollouts ----------
model.eval()
learned_policy_fn = learned_policy_fn_factory(model)

# Start slightly off center to trigger corrections
x0_test = rng.normal(0.0, TEST_PERTURB_STD)
print("Test initial state x0:", x0_test)

expert_xs, expert_us = rollout_with_policy(expert_policy_fn, x0_test, horizon=50)
learned_xs, learned_us = rollout_with_policy(learned_policy_fn, x0_test, horizon=50)

print("Expert max |x|:", np.max(np.abs(expert_xs)))
print("Learned max |x|:", np.max(np.abs(learned_xs)))
