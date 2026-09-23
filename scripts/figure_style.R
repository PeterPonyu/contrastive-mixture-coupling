library(ggplot2)
library(grid)

paper_theme <- function(base_size = 10) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      text = element_text(family = "Arial", colour = "black", face = "plain"),
      axis.text = element_text(colour = "black"),
      axis.title = element_text(colour = "black", face = "plain"),
      plot.title = element_text(hjust = 0.5, face = "plain", size = base_size + 1,
                                margin = margin(b = 4)),
      plot.subtitle = element_text(hjust = 0.5, face = "plain"),
      plot.tag = element_text(face = "bold", colour = "black", size = base_size + 2),
      plot.tag.position = c(0.015, 0.965),
      plot.background = element_rect(fill = "transparent", colour = NA),
      panel.background = element_rect(fill = "transparent", colour = NA),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.box.just = "center",
      legend.spacing.x = unit(3, "mm"),
      legend.key.width = unit(5, "mm"),
      legend.key.height = unit(4, "mm"),
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.box.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA),
      legend.title = element_text(face = "plain", colour = "black"),
      legend.text = element_text(colour = "black", face = "plain"),
      strip.background = element_blank(),
      strip.text = element_text(face = "plain", colour = "black"),
      plot.margin = margin(5, 5, 4, 5)
    )
}

save_panels <- function(plots, filename, ncol = 2, width = 7.2,
                        height = 2.9 * ceiling(length(plots) / ncol), tags = length(plots) > 1) {
  measure_file <- tempfile(fileext = ".pdf")
  cairo_pdf(measure_file, width = width, height = height, family = "Arial")
  on.exit({ dev.off(); unlink(measure_file) }, add = TRUE)
  styled <- lapply(seq_along(plots), function(index) {
    panel <- plots[[index]] + paper_theme()
    if (tags) panel <- panel + labs(tag = LETTERS[index])
    ggplotGrob(panel)
  })
  figure <- gridExtra::arrangeGrob(grobs = styled, ncol = ncol)
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  ggsave(filename, figure, device = cairo_pdf, width = width, height = height,
         units = "in", bg = "transparent", family = "Arial", limitsize = FALSE)
  invisible(figure)
}
