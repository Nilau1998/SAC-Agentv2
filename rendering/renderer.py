import pygame
import os
import math

RAD_TO_DEG = 180 / 3.14159


class RenderBoatEnv:
    metadata = {"render.modes": ["human"], "render_fps": 30}

    def __init__(self, config):
        self.config = config
        self.state = None
        self.wind_strength = None
        self.wind_angle = None

        self.window_size_x = self.config.pygame.screen_width
        self.window_size_y = self.config.pygame.screen_height
        self.pix_square_size_x = self.window_size_x / self.config.boat_env.goal_line
        self.pix_square_size_y = self.window_size_y / self.config.boat_env.track_width

        self.boat_image = pygame.image.load(
            os.path.join("rendering", "assets", "boat_topdown.png")
        )

        self.window = None
        self.clock = None
        self.setup()

    def setup(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size_x, self.window_size_y)
            )
            self.window.fill((255, 255, 255))
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        # Draw water
        self._gradient_rect(
            self.window,
            (255, 255, 255),
            (0, 128, 255),
            pygame.Rect(0, 0, self.window_size_x, self.window_size_y / 2),
        )
        self._gradient_rect(
            self.window,
            (0, 128, 255),
            (255, 255, 255),
            pygame.Rect(
                0, self.window_size_y / 2, self.window_size_x, self.window_size_y / 2
            ),
        )

        # Draw goal line
        pygame.draw.line(
            self.window,
            (0, 153, 0),
            (self.config.boat_env.goal_line * self.pix_square_size_x, 0),
            (
                self.config.boat_env.goal_line * self.pix_square_size_x,
                self.window_size_y,
            ),
            5,
        )

        # Draw boat
        self._draw_boat()

        # Draw label
        self._draw_label()

        # Draw wind
        self._draw_wind()

        pygame.display.update()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def set_state(self, state):
        self.state = state

    def set_wind(self, wind):
        self.wind_strength = wind[0]
        self.wind_angle = wind[1]

    def _draw_boat(self):
        if self.state is not None:
            rotated_image = pygame.transform.rotate(
                self.boat_image, (-self.state["boat_angle"] * RAD_TO_DEG) - 90
            )
            scaled_image = pygame.transform.scale(
                rotated_image,
                (50, 50),
            )
            self.window.blit(
                scaled_image,
                (
                    self.state["boat_position_x"] * self.pix_square_size_x,
                    self.window_size_y / 2
                    + self.state["boat_position_y"] * self.pix_square_size_y,
                ),
            )

    def _draw_label(self):
        font = pygame.font.Font(None, 36)

        text = font.render("Boat Dynamics", True, (0, 0, 0))
        self.window.blit(text, (10, 10))

        font = pygame.font.Font(None, 20)

        text = font.render(
            f"Position: ({self.state['boat_position_x']:.2f}, {-self.state['boat_position_y']:.2f})",
            True,
            (0, 0, 0),
        )
        self.window.blit(text, (10, 35))

        text = font.render(
            f"Angle: {(self.state['boat_angle'] * RAD_TO_DEG):.2f}", True, (0, 0, 0)
        )
        self.window.blit(text, (10, 50))

        text = font.render(f"Reward: {self.state['reward']:.2f}", True, (0, 0, 0))
        self.window.blit(text, (10, 65))

    def _draw_wind(self):
        if self.wind_strength is not None and self.wind_angle is not None:
            x = 100
            y = self.window_size_y - 100
            # Circle
            pygame.draw.circle(self.window, (0, 0, 0), (x, y), 52)
            pygame.draw.circle(self.window, (211, 211, 211), (x, y), 50)

            # Direction
            pygame.draw.line(
                self.window,
                (255, 0, 0),
                (x, y),
                (
                    x + math.cos(self.wind_angle) * 50,
                    y + math.sin(self.wind_angle) * 50,
                ),
                2,
            )

            # Strength
            pygame.draw.rect(
                self.window, (0, 0, 0), pygame.Rect(x + 54, y - 53, 21, 105)
            )
            pygame.draw.rect(
                self.window, (211, 211, 211), pygame.Rect(x + 56, y - 51, 17, 101)
            )
            pygame.draw.rect(
                self.window,
                (255, 0, 0),
                pygame.Rect(x + 56, y - 51, 17, 101 * (1 - self.wind_strength)),
            )

    def _gradient_rect(self, window, color_a, color_b, target_rect):
        color_rect = pygame.Surface((2, 2))
        pygame.draw.line(color_rect, color_a, (0, 0), (1, 0))
        pygame.draw.line(color_rect, color_b, (0, 1), (1, 1))
        color_rect = pygame.transform.smoothscale(
            color_rect, (target_rect.width, target_rect.height)
        )
        window.blit(color_rect, target_rect)
