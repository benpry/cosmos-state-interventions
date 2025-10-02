import time

import gymnasium as gym
import numpy as np
import pygame


class FrozenLakeGUI:
    def __init__(self, map_name="4x4", is_slippery=False, agent_policy=None):
        # Initialize environment with rgb_array rendering
        self.env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="rgb_array",
        )
        self.state, _ = self.env.reset()

        # Agent policy (function that takes state and returns action)
        self.agent_policy = agent_policy or self.random_policy

        # Initialize Pygame
        pygame.init()

        # Get map dimensions
        self.desc = self.env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

        # Render once to get the image size
        rendered_img = self.env.render()
        self.render_height, self.render_width = rendered_img.shape[:2]

        # Calculate cell size from rendered image
        self.cell_size = self.render_width // self.ncol

        # Set up display with info panel
        self.info_height = 120
        self.width = self.render_width
        self.height = self.render_height + self.info_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Frozen Lake - RL Agent with Manual Intervention")

        # Colors for info panel
        self.colors = {
            "bg": (240, 240, 240),
            "text": (0, 0, 0),
        }

        # Font
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        # Game state
        self.running = True
        self.paused = False
        self.done = False
        self.total_reward = 0
        self.steps = 0
        self.agent_step_delay = 0.5  # seconds between agent actions
        self.last_step_time = time.time()

        # Dragging state
        self.dragging = False
        self.drag_start_state = None

    def random_policy(self, state):
        """Default random policy"""
        return self.env.action_space.sample()

    def get_agent_pos(self):
        """Convert state to (row, col) position"""
        return divmod(self.state, self.ncol)

    def pos_to_state(self, row, col):
        """Convert (row, col) to state number"""
        return row * self.ncol + col

    def get_cell_from_pixel(self, x, y):
        """Convert pixel coordinates to grid cell (row, col)"""
        if y >= self.render_height:
            return None, None
        col = x // self.cell_size
        row = y // self.cell_size
        if 0 <= row < self.nrow and 0 <= col < self.ncol:
            return row, col
        return None, None

    def is_cell_clicked(self, mouse_pos):
        """Check if a grid cell was clicked and return its row, col"""
        return self.get_cell_from_pixel(*mouse_pos)

    def set_agent_state(self, new_state):
        """Manually set the agent's state"""
        self.state = new_state
        # Update the environment's internal state
        self.env.unwrapped.s = new_state

        # Check if the new state is terminal
        row, col = self.get_agent_pos()
        cell = self.desc[row, col].decode("utf-8")
        if cell == "H":
            self.done = True
        elif cell == "G":
            self.done = True
            self.total_reward += 1

    def draw_environment(self):
        """Draw the frozen lake using gymnasium's built-in renderer"""
        # Get the rendered frame from the environment
        rendered_img = self.env.render()

        # Convert RGB array to pygame surface
        # Transpose because pygame expects (width, height, 3) but numpy gives (height, width, 3)
        surf = pygame.surfarray.make_surface(np.transpose(rendered_img, (1, 0, 2)))

        # Blit to screen
        self.screen.blit(surf, (0, 0))

        # Draw a highlight if dragging
        if self.dragging:
            mouse_pos = pygame.mouse.get_pos()
            row, col = self.get_cell_from_pixel(*mouse_pos)
            if row is not None and col is not None:
                x = col * self.cell_size
                y = row * self.cell_size
                # Draw semi-transparent highlight
                highlight = pygame.Surface((self.cell_size, self.cell_size))
                highlight.set_alpha(100)
                highlight.fill((255, 255, 0))  # Yellow highlight
                self.screen.blit(highlight, (x, y))

    def draw_info(self):
        """Draw information panel"""
        info_y = self.render_height
        pygame.draw.rect(
            self.screen, self.colors["bg"], (0, info_y, self.width, self.info_height)
        )

        # Status text
        if self.done:
            if self.total_reward > 0:
                status = "SUCCESS! Press R to restart"
                color = (0, 150, 0)
            else:
                status = "FAILED! Press R to restart"
                color = (150, 0, 0)
        elif self.paused:
            status = "PAUSED - Press SPACE to resume"
            color = (150, 100, 0)
        else:
            status = "Agent Running - Click to intervene"
            color = self.colors["text"]

        status_text = self.font.render(status, True, color)
        status_rect = status_text.get_rect(center=(self.width // 2, info_y + 25))
        self.screen.blit(status_text, status_rect)

        # Stats
        stats = f"Steps: {self.steps}  Reward: {self.total_reward:.1f}"
        stats_text = self.small_font.render(stats, True, self.colors["text"])
        stats_rect = stats_text.get_rect(center=(self.width // 2, info_y + 60))
        self.screen.blit(stats_text, stats_rect)

        # Controls
        controls = "R: Reset | SPACE: Pause | Click: Move Agent"
        controls_text = self.small_font.render(controls, True, self.colors["text"])
        controls_rect = controls_text.get_rect(center=(self.width // 2, info_y + 90))
        self.screen.blit(controls_text, controls_rect)

    def handle_input(self):
        """Handle keyboard and mouse input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset
                    self.state, _ = self.env.reset()
                    self.done = False
                    self.total_reward = 0
                    self.steps = 0
                    self.paused = False
                    self.last_step_time = time.time()

                elif event.key == pygame.K_SPACE:
                    # Toggle pause
                    self.paused = not self.paused
                    if not self.paused:
                        self.last_step_time = time.time()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    row, col = self.is_cell_clicked(event.pos)
                    if row is not None and col is not None:
                        self.dragging = True
                        self.drag_start_state = self.state
                        self.paused = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.dragging:
                    # Drop the agent
                    self.dragging = False
                    mouse_pos = pygame.mouse.get_pos()
                    row, col = self.get_cell_from_pixel(*mouse_pos)

                    if row is not None and col is not None:
                        new_state = self.pos_to_state(row, col)
                        self.set_agent_state(new_state)

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    # Just for visual feedback, actual move happens on mouse up
                    pass

    def agent_step(self):
        """Let the agent take one step"""
        if not self.done and not self.paused and not self.dragging:
            current_time = time.time()
            if current_time - self.last_step_time >= self.agent_step_delay:
                # Get action from policy
                action = self.agent_policy(self.state)

                # Take action
                self.state, reward, terminated, truncated, _ = self.env.step(action)
                self.done = terminated or truncated
                self.total_reward += reward
                self.steps += 1
                self.last_step_time = current_time

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()

        while self.running:
            self.handle_input()
            self.agent_step()

            # Draw
            self.screen.fill(self.colors["bg"])
            self.draw_environment()
            self.draw_info()

            pygame.display.flip()
            clock.tick(60)  # 60 FPS

        pygame.quit()
        self.env.close()


def example_trained_policy(state):
    """Example: A simple policy that tries to go right then down"""
    # This is just a demonstration - replace with your actual RL policy
    # For a 4x4 grid, this policy tries to reach the goal
    policy_map = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,  # Row 0: go down
        4: 2,
        5: 1,
        6: 1,
        7: 1,  # Row 1: first cell right, others down
        8: 2,
        9: 2,
        10: 1,
        11: 1,  # Row 2: go right/down
        12: 2,
        13: 2,
        14: 2,
        15: 2,  # Row 3: go right
    }
    return policy_map.get(state, 0)


if __name__ == "__main__":
    # Create and run the GUI
    # You can pass your own policy function here
    game = FrozenLakeGUI(
        map_name="4x4",
        is_slippery=False,
        agent_policy=example_trained_policy,  # Replace with your RL agent
    )
    game.run()
