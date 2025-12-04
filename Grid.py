from typing import Iterable, Optional, Tuple, List
import pygame  # Import only when needed for rendering

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
        extractionTL_x: int,
        extractionTL_y: int,
        extractionBR_x: int,
        extractionBR_y: int,
        visibilityRadius: int = Agent.visibilityRadius,
        width: int = gridWidth,
        height: int = gridHeight,
    ):
        self.width = width
        self.height = height

        self.agentCoords: List[Tuple[int, int]] = []
        self.ghostCoords: Optional[Tuple[int, int]] = None

        self.extraction_area_tl: Tuple[int, int] = (extractionTL_x, extractionTL_y)
        self.extraction_area_br: Tuple[int, int] = (extractionBR_x, extractionBR_y)
        self.extraction_point_center = ((extractionTL_x + extractionBR_x) / 2.0, (extractionTL_y + extractionBR_y) / 2.0)
        self.visibilityRadius = visibilityRadius

        # Rendering state (pygame imported lazily only when needed)
        self.cell_size: int = 20
        self.surface: Optional[any] = None  # pygame.Surface when initialized
        self._font_small: Optional[any] = None  # pygame.font.Font when initialized


    # ---------- World helpers ----------

    def setEntities(self, agentCoords: Iterable[Tuple[int, int]], ghostCoords: Tuple[int, int]):
        self.agentCoords = list(agentCoords)
        self.ghostCoords = ghostCoords
        return self.agentCoords, self.ghostCoords

    def setSurrendered(self, surrendered: bool):
        if surrendered and self.ghostCoords is not None:
            self.ghostColor = (0,255,0)
        else: 
            self.ghostColor = (255, 255, 0)


    # ---------- Drawing ----------

    def init_display(self, cell_size: int = 20, caption: str = "Ghostbusters Grid") -> None:

        self.cell_size = cell_size
        pygame.init()
        self.surface = pygame.display.set_mode((self.width * cell_size, self.height * cell_size))
        pygame.display.set_caption(caption)
        if not pygame.font.get_init():
            pygame.font.init()

        self._font_small = pygame.font.SysFont("consolas", max(12, cell_size // 2))

    def close_display(self) -> None:
        
        if self.surface is not None:
            pygame.display.quit()
        pygame.quit()
        self.surface = None
        self._font_small = None
    
    def _cell_rect(self, x: int, y: int):
        
        s = self.cell_size
        return pygame.Rect(x * s, y * s, s, s)

    def draw_grid(self, show_grid_lines: bool = True) -> None:
        
        if self.surface is None:
            self.init_display()

        surf = self.surface
        s = self.cell_size

        # Background
        surf.fill(bkgColor)

        # Vision shading 
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
        tlx, tly = self.extraction_area_tl
        brx, bry = self.extraction_area_br

        # Ensure proper ordering and clip to grid bounds
        xmin, xmax = sorted((tlx, brx))
        ymin, ymax = sorted((tly, bry))
        xmin = max(0, xmin)
        xmax = min(self.width - 1, xmax)
        ymin = max(0, ymin)
        ymax = min(self.height - 1, ymax)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                pygame.draw.rect(surf, extractionColor, self._cell_rect(x, y))

        # Agents with indices
        for i, (ax, ay) in enumerate(self.agentCoords):
            r = self._cell_rect(ax, ay)
            pygame.draw.rect(surf, agentColor, r)
            if self._font_small:
                label = self._font_small.render(str(i), True, (0, 0, 0))
                # Center the index in the cell
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

        # Small delay to fps
            pygame.time.Clock().tick(30)

        pygame.display.flip()




        

        


    

  