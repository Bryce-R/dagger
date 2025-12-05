# ---------- Generate data ----------
contexts, targets = generate_dataset(NUM_TRAJ)

# Split train/val
num_samples = contexts.shape[0]
idx = np.arange(num_samples)
rng.shuffle(idx)
split = int(0.9 * num_samples)
train_idx, val_idx = idx[:split], idx[split:]

train_ds = ImitationDataset(contexts[train_idx], targets[train_idx])
val_ds   = ImitationDataset(contexts[val_idx], targets[val_idx])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)

# ---------- Model ----------
model = TinyTransformerPolicy().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

# ---------- Training ----------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x_seq, u in train_loader:
        x_seq = x_seq.to(device)          # (B, T, 1)
        u = u.to(device)                  # (B,)

        pred_u = model(x_seq)             # (B,)
        loss = criterion(pred_u, u)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

    avg_train_loss = total_loss / len(train_ds)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_seq, u in val_loader:
            x_seq = x_seq.to(device)
            u = u.to(device)
            pred_u = model(x_seq)
            loss = criterion(pred_u, u)
            val_loss += loss.item() * x_seq.size(0)

    avg_val_loss = val_loss / len(val_ds)
    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
