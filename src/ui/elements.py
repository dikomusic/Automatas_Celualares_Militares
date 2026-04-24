import pygame

class Button:
    def __init__(self, x, y, w, h, text, color=(100, 100, 100)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.active = False

    def draw(self, screen):
        # Efecto de resaltado si está activo
        draw_color = (200, 150, 50) if self.active else self.color
        pygame.draw.rect(screen, draw_color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

class Dashboard:
    def __init__(self, x, width, height):
        self.rect = pygame.Rect(x, 0, width, height)
        self.font = pygame.font.SysFont("Consolas", 16)

    def draw(self, screen, stats):
        # Fondo del panel
        pygame.draw.rect(screen, (30, 30, 35), self.rect)
        pygame.draw.line(screen, (100, 100, 100), (self.rect.x, 0), (self.rect.x, self.rect.height), 3)

        # Monitor de Resultados en Tiempo Real
        y_offset = 20
        title = self.font.render("MONITOR TÁCTICO", True, (200, 200, 0))
        screen.blit(title, (self.rect.x + 20, y_offset))
        
        y_offset += 40
        for label, value in stats.items():
            stat_surf = self.font.render(f"{label}: {value}", True, (255, 255, 255))
            screen.blit(stat_surf, (self.rect.x + 20, y_offset))
            y_offset += 25
