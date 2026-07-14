from src.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

print("Train:", len(train_loader.dataset))
print("Validation:", len(val_loader.dataset))
print("Test:", len(test_loader.dataset))

print()

print(train_loader.dataset.class_distribution())