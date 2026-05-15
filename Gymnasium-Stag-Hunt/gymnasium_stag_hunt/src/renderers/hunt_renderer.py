import pygame as pg

from gymnasium_stag_hunt.src.entities import Entity, TILE_SIZE
from gymnasium_stag_hunt.src.renderers.abstract_renderer import AbstractRenderer

# Gold highlight when stag is caught (both agents on stag)
STAG_CAUGHT_COLOR = (255, 215, 0)
STAG_CAUGHT_BORDER = 4


class HuntRenderer(AbstractRenderer):
    def __init__(self, game, window_title, screen_size):
        super(HuntRenderer, self).__init__(
            game=game, window_title=window_title, screen_size=screen_size
        )

        entity_positions = self._game.ENTITY_POSITIONS

        self._stag_sprite = Entity(
            entity_type="stag", location=entity_positions["stag"]
        )
        self._plant_sprites = self._make_plant_entities(entity_positions["plants"])
        self._capture_flash = None  # pixel (x, y) of capture cell, if any

        self._draw_grid()

    """
    Misc
    """

    def _make_plant_entities(self, locations):
        plants = []
        for loc in locations:
            plants.append(Entity(entity_type="plant", location=loc))
        return plants

    def _draw_entities(self):
        """
        Plants first, then stag, then agents. If a capture just happened, draw
        a gold border + ghost stag on the capture cell so the success is
        visible (the real stag has already teleported away).
        """
        stag_pos = tuple(self._stag_sprite.rect.topleft)
        a_pos = tuple(self._a_sprite.rect.topleft)
        b_pos = tuple(self._b_sprite.rect.topleft)

        # Plants
        for plant in self._plant_sprites:
            self._entity_layer.blit(plant.IMAGE, (plant.rect.left, plant.rect.top))

        # Capture flash: hide newly-spawned stag, show capture cell only
        if self._capture_flash is not None:
            fx, fy = self._capture_flash
            pg.draw.rect(
                self._entity_layer,
                STAG_CAUGHT_COLOR,
                pg.Rect(fx, fy, TILE_SIZE, TILE_SIZE),
                STAG_CAUGHT_BORDER,
            )
            half = TILE_SIZE // 2
            stag_small = pg.transform.scale(self._stag_sprite.IMAGE, (TILE_SIZE, half))
            a_small = pg.transform.scale(self._a_sprite.IMAGE, (half, half))
            b_small = pg.transform.scale(self._b_sprite.IMAGE, (half, half))
            self._entity_layer.blit(stag_small, (fx, fy))
            self._entity_layer.blit(a_small, (fx, fy + half))
            self._entity_layer.blit(b_small, (fx + half, fy + half))
            return

        # Stag at current position
        self._entity_layer.blit(
            self._stag_sprite.IMAGE,
            (self._stag_sprite.rect.left, self._stag_sprite.rect.top),
        )

        # Agents overlap same cell (no capture): split tile so both visible
        if a_pos == b_pos:
            half = TILE_SIZE // 2
            a_small = pg.transform.scale(self._a_sprite.IMAGE, (half, TILE_SIZE))
            b_small = pg.transform.scale(self._b_sprite.IMAGE, (half, TILE_SIZE))
            self._entity_layer.blit(a_small, (a_pos[0], a_pos[1]))
            self._entity_layer.blit(b_small, (a_pos[0] + half, a_pos[1]))
            return

        # Single agent on stag: shrink so stag stays visible underneath
        if a_pos == stag_pos:
            half = TILE_SIZE // 2
            a_small = pg.transform.scale(self._a_sprite.IMAGE, (half, half))
            self._entity_layer.blit(a_small, (a_pos[0], a_pos[1] + half))
        else:
            self._entity_layer.blit(
                self._a_sprite.IMAGE,
                (self._a_sprite.rect.left, self._a_sprite.rect.top),
            )
        if b_pos == stag_pos:
            half = TILE_SIZE // 2
            b_small = pg.transform.scale(self._b_sprite.IMAGE, (half, half))
            self._entity_layer.blit(b_small, (b_pos[0] + half, b_pos[1] + half))
        else:
            self._entity_layer.blit(
                self._b_sprite.IMAGE,
                (self._b_sprite.rect.left, self._b_sprite.rect.top),
            )

    def _update_rects(self, entity_positions):
        self._a_sprite.update_rect(entity_positions["a_agent"])
        self._b_sprite.update_rect(entity_positions["b_agent"])
        self._stag_sprite.update_rect(entity_positions["stag"])
        plants_pos = entity_positions["plants"]
        idx = 0
        for plant in self._plant_sprites:
            plant.update_rect(plants_pos[idx])
            idx = idx + 1

        # Convert grid-cell capture position to pixel coords
        cap = entity_positions.get("capture_flash")
        if cap is None:
            self._capture_flash = None
        else:
            self._capture_flash = (cap[0] * TILE_SIZE, cap[1] * TILE_SIZE)
