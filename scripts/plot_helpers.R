suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(jsonlite)
})
source(file.path(root, "scripts", "figure_style.R"))
seed_colours <- c("0" = "#27647a", "1" = "#d48638", "2" = "#76519a")
seed_shapes <- c("0" = 16, "1" = 15, "2" = 18)
backgrounds <- c("setty", "dentate", "endo", "lung")
background_labels <- c(setty = "Setty", dentate = "Dentate", endo = "Endo", lung = "Lung")
figure_counts <- list()
dir.create(file.path(root, "figures"), recursive = TRUE, showWarnings = FALSE)
read_evidence <- function(filename) {
  source_text <- paste(readLines(file.path(root, "evidence", filename), warn = FALSE), collapse = "\n")
  source_text <- gsub("(?<=[: ,\\[\\n])(?:NaN|-?Infinity)(?=[,}\\] \\n])", "null", source_text, perl = TRUE)
  fromJSON(source_text, simplifyVector = FALSE)
}
rows_frame <- function(rows, fields) {
  as.data.frame(setNames(lapply(fields, function(field) {
    vapply(rows, function(row) {
      value <- row
      for (key in strsplit(field, ".", fixed = TRUE)[[1]]) value <- value[[key]]
      if (is.null(value)) NA_character_ else as.character(value)
    }, character(1))
  }), names(fields)), stringsAsFactors = FALSE)
}
numeric_columns <- function(data, columns) {
  for (column in columns) data[[column]] <- as.numeric(data[[column]])
  data
}
read_sweep <- function(directory) {
  rows <- read_evidence(file.path(directory, "sweep_scores.json"))$runs
  data <- rows_frame(rows, c(dataset = "dataset", seed = "seed", axis = "axis", value = "value",
    edge = "edge_survival", branch = "branch_knn_mean_r2", occupancy = "mixture_effective_components",
    coherence = "allocation_coherence", perplexity = "allocation_perplexity_mean",
    separation = "separation_survival", reachability = "reachability_survival"))
  numeric_columns(data, c("value", "edge", "branch", "occupancy", "coherence", "perplexity", "separation", "reachability"))
}
seed_scales <- function() {
  list(scale_colour_manual(values = seed_colours, name = "Model seed"),
       scale_shape_manual(values = seed_shapes, name = "Model seed"))
}
save_figure <- function(plots, filename, ncol = 2, width = 7.2,
                        height = 2.48 * ceiling(length(plots) / ncol), point_counts,
                        row_labels = NULL) {
  scratch <- tempfile(fileext = ".pdf")
  cairo_pdf(scratch, width = width, height = height, family = "Arial")
  on.exit({dev.off(); unlink(scratch)}, add = TRUE)
  styled <- lapply(seq_along(plots), function(index) {
    panel <- plots[[index]] + paper_theme()
    if (length(plots) > 1 && is.null(row_labels)) panel <- panel + labs(tag = LETTERS[index])
    panel
  })
  first <- ggplotGrob(styled[[1]])
  guide_indices <- which(first$layout$name %in% c("guide-box", "guide-box-bottom"))
  legend <- if (length(guide_indices)) first$grobs[[guide_indices[[1]]]] else NULL
  panels <- lapply(styled, function(panel) {
    ggplotGrob(panel + theme(legend.position = "none"))
  })
  common_widths <- do.call(unit.pmax, lapply(panels, function(panel) panel$widths))
  panels <- lapply(panels, function(panel) { panel$widths <- common_widths; panel })
  rows <- split(seq_along(panels), ceiling(seq_along(panels) / ncol))
  pieces <- list()
  heights <- list()
  has_legend <- !is.null(legend) && !inherits(legend, "zeroGrob")
  for (row_index in seq_along(rows)) {
    if (!is.null(row_labels)) {
      pieces[[length(pieces) + 1]] <- grobTree(
        textGrob(LETTERS[row_index], x = unit(4, "mm"), just = "left",
                 gp = gpar(fontfamily = "Arial", fontface = "bold", fontsize = 12)),
        textGrob(row_labels[row_index],
                 gp = gpar(fontfamily = "Arial", col = "black", fontsize = 11)))
      heights[[length(heights) + 1]] <- unit(5, "mm")
    }
    pieces[[length(pieces) + 1]] <- arrangeGrob(grobs = panels[rows[[row_index]]], ncol = ncol)
    heights[[length(heights) + 1]] <- unit(1, "null")
    if (has_legend && row_index == 1) {
      pieces[[length(pieces) + 1]] <- legend
      heights[[length(heights) + 1]] <- sum(legend$heights) + unit(1, "mm")
    }
  }
  figure <- arrangeGrob(grobs = pieces, ncol = 1, heights = do.call(unit.c, heights))
  ggsave(file.path(root, "figures", paste0(filename, ".pdf")), figure,
         device = cairo_pdf, width = width, height = height,
         bg = "transparent", family = "Arial", limitsize = FALSE)
  ggsave(file.path(root, "figures", paste0(filename, ".png")), figure,
         device = grDevices::png, type = "cairo", width = width, height = height,
         dpi = 160, bg = "transparent", limitsize = FALSE)
  figure_counts[[filename]] <<- list(
    points_per_panel = point_counts,
    lettering = if (is.null(row_labels)) "one letter per distinct panel" else setNames(row_labels, LETTERS[seq_along(row_labels)]),
    legend = if (!has_legend) "none" else if (length(rows) > 1) "one horizontal full-width inter-row gutter" else "one horizontal full-width footer")
}
paired_panels <- function(data, field, ylabel, filename, arrows = FALSE) {
  chosen <- subset(data, axis == "mixture_weight" & value %in% c(0, 1))
  limits <- range(chosen[[field]], na.rm = TRUE)
  plots <- lapply(backgrounds, function(background) {
    panel <- subset(chosen, dataset == background)
    panel$x <- panel$value + (as.numeric(panel$seed) - 1) * 0.065
    panel$score <- panel[[field]]
    panel <- panel[order(panel$seed, panel$value), ]
    ggplot(panel, aes(x, score, colour = seed, shape = seed, group = seed)) +
      geom_path(linewidth = 0.5, arrow = if (arrows) arrow(length = unit(1.4, "mm"), type = "closed") else NULL) +
      geom_point(size = 2) + seed_scales() +
      scale_x_continuous(breaks = c(0, 1), labels = c("w = 0", "w = 1"), limits = c(-0.2, 1.2)) +
      scale_y_continuous(limits = limits + c(-1, 1) * max(diff(limits) * 0.08, 0.001)) +
      labs(x = "Mixture coupling", y = ylabel, title = background_labels[[background]])
  })
  save_figure(plots, filename, point_counts = rep(6, 4))
}
plot_umap <- function() {
  files <- sort(list.files(file.path(root, "evidence", "umap"), pattern = "\\.csv$", full.names = TRUE))
  stopifnot(length(files) == 6)
  frames <- lapply(files, read.csv, stringsAsFactors = FALSE)
  stopifnot(all(vapply(frames, nrow, integer(1)) == 450))
  data <- do.call(rbind, frames)
  palette <- c("0" = "#1f4e79", "1" = "#2e7d4f", "2" = "#b5473f", "3" = "#8a4b08",
               "4" = "#7b2d8e", "5" = "#0e7c7b", "6" = "#c27c0e", "7" = "#4a6fa5",
               "8" = "#6b6b6b", "9" = "#a34d7a")
  plots <- list()
  for (coupling in c("w=0", "w=1")) for (seed in 0:2) {
    panel <- data[data$dose == coupling & data$seed == seed, ]
    stopifnot(nrow(panel) == 450)
    panel$label <- factor(panel$label, levels = 0:9)
    span <- max(diff(range(panel$x)), diff(range(panel$y))) * 1.08
    x_limits <- mean(range(panel$x)) + c(-0.5, 0.5) * span
    y_limits <- mean(range(panel$y)) + c(-0.5, 0.5) * span
    plots[[length(plots) + 1]] <- ggplot(panel, aes(x, y, colour = label)) +
      geom_point(size = 0.65, alpha = 0.85, stroke = 0) +
      scale_colour_manual(values = palette, drop = FALSE, name = "Stored label") +
      coord_equal(xlim = x_limits, ylim = y_limits, expand = FALSE) +
      labs(x = "UMAP 1", y = "UMAP 2", title = paste("Model seed", seed)) +
      guides(colour = guide_legend(nrow = 1, title.position = "left",
                                   override.aes = list(size = 1.6, alpha = 1)))
  }
  save_figure(plots, "setty_umap", ncol = 3, width = 8.4, height = 5.8,
              point_counts = rep(450, 6), row_labels = c("Uncoupled: w = 0", "Coupled: w = 1"))
}
write_figure_manifest <- function() {
  write_json(list(generator = "R: ggplot2/grid/Cairo", font = "Arial",
                  figures = figure_counts), file.path(root, "figures", "plot_manifest.json"),
             pretty = TRUE, auto_unbox = TRUE)
}
