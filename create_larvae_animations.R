#!/usr/bin/env Rscript

# Create larval trajectory animations from tracking data.
# Outputs (when MP4 encoder available):
#   animations/larvae_tracks_window.mp4
#   animations/larvae_density.mp4
#
# If MP4 encoding is unavailable in this R environment, the script falls back to:
#   animations/larvae_tracks_window.gif
#   animations/larvae_density.gif

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

# =========================
# Config (edit as needed)
# =========================
DATA_PATH <- "embedding_dan.csv"
OUTPUT_DIR <- "animations"

# Optional fixed column names. Use NULL for auto-detection.
X_COLUMN_OVERRIDE <- NULL
Y_COLUMN_OVERRIDE <- NULL
FRAME_COLUMN_OVERRIDE <- NULL
ID_COLUMN_OVERRIDE <- NULL

# Optional filter by group condition/category.
GROUP_CONDITION_COLUMN <- "condition"
GROUP_CONDITION_VALUE <- NULL

WINDOW_SIZE_FRAMES <- 50
FRAME_STRIDE <- 1
RECORDING_FPS <- 16
FPS <- 16
WIDTH_PX <- 1080
HEIGHT_PX <- 1080
DPI <- 140

MAKE_TRACK_ANIMATION <- TRUE
MAKE_DENSITY_ANIMATION <- TRUE

DENSITY_BINS <- 90
DENSITY_METHOD <- "kde" # "hist" or "kde"
KDE_BANDWIDTH <- 1.2

TEST_DURATION_SECONDS <- 30


pick_first_existing <- function(candidates, columns) {
  for (c in candidates) {
    if (c %in% columns) return(c)
  }
  NULL
}

detect_columns <- function(columns) {
  frame_col <- if (!is.null(FRAME_COLUMN_OVERRIDE)) FRAME_COLUMN_OVERRIDE else
    pick_first_existing(c("frame", "t", "time", "timestamp", "frame_idx"), columns)
  id_col <- if (!is.null(ID_COLUMN_OVERRIDE)) ID_COLUMN_OVERRIDE else
    pick_first_existing(c("id", "track_id", "larva_id", "track", "animal_id"), columns)
  x_col <- if (!is.null(X_COLUMN_OVERRIDE)) X_COLUMN_OVERRIDE else
    pick_first_existing(c("centroid_x", "centoid_x", "spinepoint_x6_conv", "spinepoint_x_6_conv"), columns)
  y_col <- if (!is.null(Y_COLUMN_OVERRIDE)) Y_COLUMN_OVERRIDE else
    pick_first_existing(c("centroid_y", "centoid_y", "spinepoint_y6_conv", "spinepoint_y_6_conv"), columns)

  missing <- c()
  if (is.null(frame_col)) missing <- c(missing, "frame/time column (e.g. frame, t, time)")
  if (is.null(id_col)) missing <- c(missing, "larva ID column (e.g. id, track_id)")
  if (is.null(x_col)) {
    missing <- c(
      missing,
      "x position column (tried: centroid_x, centoid_x, spinepoint_x6_conv, spinepoint_x_6_conv)"
    )
  }
  if (is.null(y_col)) {
    missing <- c(
      missing,
      "y position column (tried: centroid_y, centoid_y, spinepoint_y6_conv, spinepoint_y_6_conv)"
    )
  }
  if (length(missing) > 0) {
    stop(
      paste0(
        "Required columns missing:\n- ",
        paste(missing, collapse = "\n- "),
        "\n\nAvailable columns:\n",
        paste(columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  message("Detected columns:")
  message("  frame/time: ", frame_col)
  message("  larva ID:    ", id_col)
  message("  x position:  ", x_col)
  message("  y position:  ", y_col)

  list(frame = frame_col, id = id_col, x = x_col, y = y_col)
}

stabilized_limits <- function(min_val, max_val) {
  if (isTRUE(all.equal(min_val, max_val))) {
    pad <- if (isTRUE(all.equal(min_val, 0))) 1.0 else abs(min_val) * 0.05
    return(c(min_val - pad, max_val + pad))
  }
  c(min_val, max_val)
}

ensure_output_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
}

encode_video <- function(frame_paths, output_mp4, fps) {
  # Prefer MP4 via av package. If unavailable, fallback to GIF via gifski.
  if (requireNamespace("av", quietly = TRUE)) {
    av::av_encode_video(frame_paths, output = output_mp4, framerate = fps, vfilter = "format=yuv420p")
    return(output_mp4)
  }
  if (requireNamespace("gifski", quietly = TRUE)) {
    output_gif <- sub("\\.mp4$", ".gif", output_mp4)
    gifski::gifski(frame_paths, gif_file = output_gif, width = WIDTH_PX, height = HEIGHT_PX, delay = 1 / fps)
    return(output_gif)
  }
  stop(
    "Could not encode animation. Install either `av` (preferred for MP4) or `gifski`.\n",
    "Install commands:\n",
    "  install.packages('av')\n",
    "  install.packages('gifski')",
    call. = FALSE
  )
}

compute_density_matrix <- function(window_df, x_edges, y_edges, method, bandwidth) {
  x <- window_df$x
  y <- window_df$y
  if (method == "kde") {
    if (!requireNamespace("MASS", quietly = TRUE)) {
      stop("KDE mode requires MASS package. Install with install.packages('MASS')", call. = FALSE)
    }
    kd <- MASS::kde2d(x, y, h = bandwidth, n = c(length(x_edges) - 1, length(y_edges) - 1), lims = c(range(x_edges), range(y_edges)))
    dens <- kd$z
    return(dens)
  }

  # Histogram fallback
  x_bin <- cut(x, breaks = x_edges, include.lowest = TRUE, labels = FALSE)
  y_bin <- cut(y, breaks = y_edges, include.lowest = TRUE, labels = FALSE)
  ok <- !is.na(x_bin) & !is.na(y_bin)
  if (!any(ok)) {
    return(matrix(0, nrow = length(x_edges) - 1, ncol = length(y_edges) - 1))
  }
  tab <- table(x_bin[ok], y_bin[ok])
  m <- matrix(0, nrow = length(x_edges) - 1, ncol = length(y_edges) - 1)
  m[as.integer(rownames(tab)), as.integer(colnames(tab))] <- as.numeric(tab)
  m
}

main <- function() {
  if (!file.exists(DATA_PATH)) {
    stop("Input file not found: ", normalizePath(DATA_PATH, winslash = "/", mustWork = FALSE), call. = FALSE)
  }
  ensure_output_dir(OUTPUT_DIR)

  cols <- colnames(read_csv(DATA_PATH, n_max = 0, show_col_types = FALSE))
  selection <- detect_columns(cols)

  use_cols <- c(selection$frame, selection$id, selection$x, selection$y)
  if (!is.null(GROUP_CONDITION_VALUE)) {
    if (!(GROUP_CONDITION_COLUMN %in% cols)) {
      stop("GROUP_CONDITION_COLUMN='", GROUP_CONDITION_COLUMN, "' not found in CSV columns.", call. = FALSE)
    }
    use_cols <- c(use_cols, GROUP_CONDITION_COLUMN)
  }

  df <- read_csv(DATA_PATH, col_select = all_of(unique(use_cols)), show_col_types = FALSE)
  names(df)[names(df) == selection$frame] <- "frame"
  names(df)[names(df) == selection$id] <- "larva_id"
  names(df)[names(df) == selection$x] <- "x"
  names(df)[names(df) == selection$y] <- "y"

  df <- df %>%
    mutate(
      frame = suppressWarnings(as.numeric(frame)),
      x = suppressWarnings(as.numeric(x)),
      y = suppressWarnings(as.numeric(y))
    ) %>%
    filter(!is.na(frame), !is.na(x), !is.na(y), !is.na(larva_id))

  if (!is.null(GROUP_CONDITION_VALUE)) {
    before <- nrow(df)
    df <- df %>% filter(.data[[GROUP_CONDITION_COLUMN]] == GROUP_CONDITION_VALUE)
    message(
      "Group filter: ", GROUP_CONDITION_COLUMN, " == ", GROUP_CONDITION_VALUE,
      "; kept ", format(nrow(df), big.mark = ","), "/", format(before, big.mark = ","), " rows."
    )
  }

  df <- df %>% arrange(frame, larva_id)

  if (!is.null(TEST_DURATION_SECONDS)) {
    if (TEST_DURATION_SECONDS <= 0) stop("TEST_DURATION_SECONDS must be > 0 or NULL", call. = FALSE)
    start_frame <- min(df$frame)
    end_frame <- start_frame + TEST_DURATION_SECONDS * RECORDING_FPS
    before <- nrow(df)
    df <- df %>% filter(frame <= end_frame)
    message(
      "Testing subset enabled: first ", TEST_DURATION_SECONDS, "s at ", RECORDING_FPS, " fps (<= frame ",
      round(end_frame), "); kept ", format(nrow(df), big.mark = ","), "/", format(before, big.mark = ","), " rows."
    )
  }

  xlim <- stabilized_limits(min(df$x), max(df$x))
  ylim <- stabilized_limits(min(df$y), max(df$y))
  message("Global x limits: (", round(xlim[1], 3), ", ", round(xlim[2], 3), ")")
  message("Global y limits: (", round(ylim[1], 3), ", ", round(ylim[2], 3), ")")

  frames <- sort(unique(df$frame))[seq(1, length(sort(unique(df$frame))), by = FRAME_STRIDE)]

  if (MAKE_TRACK_ANIMATION) {
    message("Rendering cumulative track animation...")
    temp_dir <- tempfile("tracks_frames_")
    dir.create(temp_dir, recursive = TRUE)
    on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

    for (i in seq_along(frames)) {
      f <- frames[i]
      frame_df <- df %>% filter(frame <= f)

      p <- ggplot(frame_df, aes(x = x, y = y, color = as.factor(larva_id))) +
        geom_point(size = 0.35, alpha = 0.35, stroke = 0) +
        coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE) +
        labs(
          title = "Larval cumulative trajectories",
          subtitle = paste0("Frame: ", as.integer(f)),
          x = "x position",
          y = "y position"
        ) +
        theme_minimal(base_size = 11) +
        theme(legend.position = "none")

      ggsave(
        filename = file.path(temp_dir, sprintf("track_%06d.png", i)),
        plot = p, width = WIDTH_PX / DPI, height = HEIGHT_PX / DPI, dpi = DPI
      )
      if (i %% 100 == 0) message("  track frame ", i, "/", length(frames))
    }

    out <- encode_video(
      list.files(temp_dir, full.names = TRUE, pattern = "\\.png$"),
      file.path(OUTPUT_DIR, "larvae_tracks_window.mp4"),
      FPS
    )
    message("Track animation complete: ", out)
  }

  if (MAKE_DENSITY_ANIMATION) {
    message("Rendering density animation (", DENSITY_METHOD, ")...")
    temp_dir <- tempfile("density_frames_")
    dir.create(temp_dir, recursive = TRUE)
    on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE)

    x_edges <- seq(xlim[1], xlim[2], length.out = DENSITY_BINS + 1)
    y_edges <- seq(ylim[1], ylim[2], length.out = DENSITY_BINS + 1)

    sample_idx <- unique(round(seq(1, length(frames), length.out = min(length(frames), 300))))
    vmax <- 1
    for (idx in sample_idx) {
      f <- frames[idx]
      window_df <- df %>% filter(frame >= (f - WINDOW_SIZE_FRAMES + 1), frame <= f)
      if (nrow(window_df) == 0) next
      dens <- compute_density_matrix(window_df, x_edges, y_edges, DENSITY_METHOD, KDE_BANDWIDTH)
      vmax <- max(vmax, max(dens, na.rm = TRUE))
    }

    for (i in seq_along(frames)) {
      f <- frames[i]
      window_df <- df %>% filter(frame >= (f - WINDOW_SIZE_FRAMES + 1), frame <= f)

      if (nrow(window_df) == 0) {
        grid_df <- expand_grid(x = x_edges[-length(x_edges)], y = y_edges[-length(y_edges)]) %>%
          mutate(d = 0)
      } else {
        dens <- compute_density_matrix(window_df, x_edges, y_edges, DENSITY_METHOD, KDE_BANDWIDTH)
        grid_df <- expand_grid(
          x = x_edges[-length(x_edges)],
          y = y_edges[-length(y_edges)]
        ) %>% mutate(d = as.vector(dens))
      }

      p <- ggplot(grid_df, aes(x = x, y = y, fill = d)) +
        geom_raster(interpolate = TRUE) +
        coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE) +
        scale_fill_viridis_c(option = "magma", limits = c(0, vmax)) +
        labs(
          title = paste0("Larval occupancy density (", toupper(DENSITY_METHOD), ", sliding window)"),
          subtitle = paste0("Frame: ", as.integer(f)),
          x = "x position",
          y = "y position",
          fill = "Density"
        ) +
        theme_minimal(base_size = 11)

      ggsave(
        filename = file.path(temp_dir, sprintf("density_%06d.png", i)),
        plot = p, width = WIDTH_PX / DPI, height = HEIGHT_PX / DPI, dpi = DPI
      )
      if (i %% 100 == 0) message("  density frame ", i, "/", length(frames))
    }

    out <- encode_video(
      list.files(temp_dir, full.names = TRUE, pattern = "\\.png$"),
      file.path(OUTPUT_DIR, "larvae_density.mp4"),
      FPS
    )
    message("Density animation complete: ", out)
  }

  message("Done.")
}

main()

