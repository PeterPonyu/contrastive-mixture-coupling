#!/usr/bin/env Rscript
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
root <- normalizePath(file.path(dirname(script), ".."))
source(file.path(root, "scripts", "plot_helpers.R"))
render_status <- system2(Sys.getenv("SVG_PYTHON", "python3"), shQuote(c(
  file.path(root, "scripts", "render_schematic.py"),
  file.path(root, "figures", "architecture.svg"),
  file.path(root, "figures", "architecture.pdf"))))
stopifnot(render_status == 0)
historical <- read_sweep("dpmm-contrastive-prior-20260907")
longer <- read_sweep("dpmm-contrastive-prior-400ep-setty-20260908")
paired_panels(historical, "edge", "Edge survival (200 epochs)", "fourbg_edge_200")
paired_panels(historical, "occupancy", "Effective component count", "fourbg_components_200")
branch <- rbind(transform(subset(historical, dataset == "setty" & axis == "mixture_weight" & value %in% c(0, 1)), epochs = 200),
                transform(subset(longer, dataset == "setty" & axis == "mixture_weight" & value %in% c(0, 1)), epochs = 400))
branch$coupling <- factor(branch$value, levels = c(0, 1), labels = c("w = 0", "w = 1"))
branch_plot <- ggplot(branch, aes(epochs, branch, colour = seed, shape = seed,
                                 linetype = coupling, group = interaction(seed, coupling))) +
  geom_hline(yintercept = 0, colour = "black", linewidth = 0.3) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.2) + seed_scales() +
  scale_linetype_manual(values = c("dashed", "solid"), name = "Coupling") +
  scale_x_continuous(breaks = c(200, 400)) +
  labs(title = "Setty branch probe across schedules", x = "Total training epochs", y = "Palantir branch-probe R²")
save_figure(list(branch_plot), "setty_palantir_seeds", ncol = 1, width = 6.8, height = 3.1, point_counts = 12)
matched <- rows_frame(read_evidence("new_results.json")$rows,
  c(seed = "seed", coupling = "w", edge = "metrics.edge_survival",
    branch = "metrics.branch_knn_mean_r2", occupancy = "training_occupancy.effective"))
matched <- numeric_columns(matched, c("coupling", "edge", "branch", "occupancy"))
stopifnot(nrow(matched) == 6)
titles <- c(edge = "Neighbourhood preservation", branch = "Annotation predictability", occupancy = "Last installed mixture")
labels <- c(edge = "Edge survival", branch = "Palantir branch-probe R²", occupancy = "Effective component count")
plots <- lapply(names(titles), function(field) {
  matched$score <- matched[[field]]
  ggplot(matched, aes(coupling, score, colour = seed, shape = seed, group = seed)) +
    geom_line(linewidth = 0.6) + geom_point(size = 2) + seed_scales() +
    scale_x_continuous(breaks = c(0, 1), labels = c("w = 0", "w = 1"), limits = c(-0.15, 1.15)) +
    labs(title = titles[[field]], x = "Mixture coupling", y = labels[[field]])
})
save_figure(plots, "matched_coupling", ncol = 3, width = 9.2, height = 3.15, point_counts = rep(6, 3))
plot_umap()
write_figure_manifest()
