#include "ggml-weight-prefetch.h"
#include "ggml-moe-backend.h"
#include "ggml-impl.h"

void ggml_weight_prefetch_plan::schedule(int n_splits, int depth) {
    entry_of_split.assign(n_splits, -1);
    issue_at_split.assign(n_splits, -1);
    for (size_t k = 0; k < entries.size(); k++) {
        entry_of_split[entries[k].split] = (int) k;
        if (k >= (size_t) depth) {
            issue_at_split[entries[k - depth].split] = (int) k;
        }
    }
}

bool ggml_weight_prefetch_eligible(
        const struct ggml_cgraph * graph,
        const struct ggml_tensor * input,
        const struct ggml_tensor * input_cpy) {
    if (graph->n_nodes == 0 || input->buffer == NULL) {
        return false;
    }

    if (ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_backend_buffer_is_host(input->buffer)) {
        return false;
    }

    const struct ggml_tensor * node = graph->nodes[0];

    if (node->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    // classic Prefill: weight is src[0]; dual Prefill: compact host weight is src[3]
    const bool classic = (node->src[0] == input_cpy);
    const bool dual    = (node->src[3] == input_cpy && node->src[3] != NULL && node->src[4] != NULL);
    if (!classic && !dual) {
        return false;
    }

    return ggml_moe_use_bulk_copy(node->src[2]->ne[1], node->src[2]->ne[0], input->ne[2], dual ? node->src[0]->ne[2] : 0);
}
