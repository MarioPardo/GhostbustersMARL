import pygame
from typing import Iterable, Optional, Tuple, List

import Agent

gridWidth = 30
gridHeight = 30

# ------------ Colors ------------
bkgColor        = (0, 0, 0)
linesColor      = (255, 255, 255)
extractionColor = (0, 0, 255)
ghostColor      = (255, 0, 0)
agentColor      = (0, 255, 0)
visionColor     = (200, 200, 200)

class Grid:
    """
    Minimal, extensible grid for the MARL ghostbusters environment.
    Owns static geometry + extraction point + basic rendering surface/state.
    Dynamic entities (agents, ghost) are held as simple tuples.
    """

    def __init__(
        self,
        extractionX: int,
        extractionY: int,
        visibilityRadius: int = Agent.visibilityRadius,
        width: int = gridWidth,
        height: int = gridHeight,
    ):
        self.width = width
        self.height = height

        self.agentCoords: List[Tuple[int, int]] = []
        self.ghostCoords: Optional[Tuple[int, int]] = None

        self.extraction_point: Tuple[int, int] = (extractionX, extractionY)
        self.visibilityRadius = visibilityRadius

        # Rendering state
        self.cell_size: int = 20
        self.surface: Optional[pygame.Surface] = None
        self._font_small: Optional[pygame.font.Font] = None


    # ---------- World helpers ----------

    def setEntities(self, agentCoords: Iterable[Tuple[int, int]], ghostCoords: Tuple[int, int]):
        self.agentCoords = list(agentCoords)
        self.ghostCoords = ghostCoords
        return self.agentCoords, self.ghostCoords

    # ---------- Drawing ----------

    def init_display(self, cell_size: int = 20, caption: str = "Ghostbusters Grid") -> None:
        """Initialize pygame window & font; call once before drawing."""
        self.cell_size = cell_size
        pygame.init()
        self.surface = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption(caption)
        if not pygame.font.get_init():
            pygame.font.init()
        # Small readable font; tweak size if you prefer
        self._font_small = pygame.font.SysFont("consolas", max(12, cell_size // 2))

    def close_display(self) -> None:
        if self.surface is not None:
            pygame.display.quit()
        pygame.quit()
        self.surface = None
        self._font_small = None
    
    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        s = self.cell_size
        return pygame.Rect(x * s, y * s, s, s)

    def draw_grid(self, show_grid_lines: bool = True) -> None:
        """Draw current grid state. Call after init_display()."""
        if self.surface is None:
            raise RuntimeError("Call init_display() before draw_grid().")

        surf = self.surface
        s = self.cell_size

        # Background
        surf.fill(bkgColor)

        # Vision shading (Chebyshev disks around agents)
        R = self.visibilityRadius
        if R > 0:
            for ax, ay in self.agentCoords:
                xmin = max(0, ax - R)
                xmax = min(self.width - 1, ax + R)
                ymin = max(0, ay - R)
                ymax = min(self.height - 1, ay + R)
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        if max(abs(x - ax), abs(y - ay)) <= R:
                            pygame.draw.rect(surf, visionColor, self._cell_rect(x, y))

        # Extraction point
        if self.extraction_point is not None:
            ex, ey = self.extraction_point
            pygame.draw.rect(surf, extractionColor, self._cell_rect(ex, ey))

        # Agents (with indices)
        for i, (ax, ay) in enumerate(self.agentCoords):
            r = self._cell_rect(ax, ay)
            pygame.draw.rect(surf, agentColor, r)
            if self._font_small:
                label = self._font_small.render(str(i), True, (0, 0, 0))
                # Center the index roughly in the cell
                lx = r.x + (s - label.get_width()) // 2
                ly = r.y + (s - label.get_height()) // 2
                surf.blit(label, (lx, ly))

        # Ghost
        if self.ghostCoords is not None:
            gx, gy = self.ghostCoords
            pygame.draw.rect(surf, ghostColor, self._cell_rect(gx, gy))

        # Grid lines
        if show_grid_lines:
            W, H = self.width * s, self.height * s
            for x in range(self.width + 1):
                pygame.draw.line(surf, linesColor, (x * s, 0), (x * s, H), 1)
            for y in range(self.height + 1):
                pygame.draw.line(surf, linesColor, (0, y * s), (W, y * s), 1)

        pygame.display.flip()




        

        


    

  