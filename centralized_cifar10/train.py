def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f}")
