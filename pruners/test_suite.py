# Prints sparsity of each model layer and the overall model sparsity
def test_sparsity(paras_to_prune):
    total_paras = 0
    zero_paras = 0

    for module, para_name in paras_to_prune:
        weight = getattr(module, para_name)
        mask = getattr(module, f"{para_name}_mask", None)

        if mask is not None:
            # Use masked weights to calculate sparsity
            total_paras += weight.numel()
            zero_paras += (weight * mask == 0).sum().item()

            layer_sparsity = (weight * mask == 0).sum().item() / weight.numel()
            print(f"Sparsity of {para_name}: {layer_sparsity:.2%}")
        else:
            print(f"No mask found for {para_name}. Skipping.")

    total_sparsity = zero_paras / total_paras
    print(f"\nOverall model sparsity: {total_sparsity:.2%}")
