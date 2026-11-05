"""Consumable quantities derived from part and tool geometry, for the user to approve.

Every figure is a geometric minimum (what covers the area or runs the length) times
(1 + an extra the user states). No supplier or shop value is invented: extras default to 0 and
the separator needs the user's own consumption figure, so it has no suggestion until given.
"""

from __future__ import annotations

from dataclasses import dataclass

from teklif.common.models import Param, Source


@dataclass(frozen=True)
class ConsumableInputs:
    skins: int  # laminate skins on the part (sandwich: 2)
    core_fire: float = 0.0  # extra core waste ratio
    film_fire: float = 0.0  # extra film adhesive waste ratio
    peel_ply_layers_per_skin: int = 1
    bag_extra: float = 0.0  # vacuum-bag pleat / overlap ratio
    tape_rows: int = 1  # sealant tape runs along the tool perimeter
    separator_l_per_m2_per_coat: float | None = None  # from the user's datasheet
    separator_coats: int = 1


@dataclass(frozen=True)
class Suggestion:
    material: str
    label: str
    qty: float | None
    unit: str
    driver: str
    formula: str
    basis: str

    def param(self, source: Source = Source.DERIVED, note: str = "") -> Param | None:
        if self.qty is None:
            return None
        return Param(
            value=self.qty,
            unit=self.unit,
            source=source,
            desc=f"{self.label}: {self.formula}",
            note=note or self.basis,
        )


def suggest(i: ConsumableInputs) -> list[Suggestion]:
    sep: float | None = None
    if i.separator_l_per_m2_per_coat is not None and i.separator_l_per_m2_per_coat > 0:
        sep = i.separator_l_per_m2_per_coat * i.separator_coats
    return [
        Suggestion(
            "KALIP_AYIRICI",
            "Kalıp ayırıcı",
            sep,
            "L/m2",
            "tool_area_m2",
            f"{i.separator_l_per_m2_per_coat} L/m² × {i.separator_coats} kat"
            if sep is not None
            else "sarfiyat (L/m² ve kat sayısı) sizden gerekir",
            "kalıp alanı (parça + flanş) üzerine sürülür; sarfiyat ürün veri sayfasından",
        ),
        Suggestion(
            "CEKIRDEK",
            "Çekirdek",
            1.0 + i.core_fire,
            "m2/m2",
            "part_area_m2",
            f"1 × (1 + {i.core_fire:g})",
            "çekirdek parça alanını bir kat kaplar; fire payı sizin girdiğiniz",
        ),
        Suggestion(
            "FILM_YAPISTIRICI",
            "Film yapıştırıcı",
            i.skins * (1.0 + i.film_fire),
            "m2/m2",
            "part_area_m2",
            f"{i.skins} yüzey × (1 + {i.film_fire:g})",
            "her kabuk ile çekirdek arasında bir film; fire payı sizin girdiğiniz",
        ),
        Suggestion(
            "PEEL_PLY",
            "Peel ply",
            float(i.skins * i.peel_ply_layers_per_skin),
            "m2/m2",
            "part_area_m2",
            f"{i.skins} yüzey × {i.peel_ply_layers_per_skin} kat",
            "her yüzeyde parça alanını kaplar",
        ),
        Suggestion(
            "AYIRICI_FILM",
            "Ayırıcı film",
            1.0,
            "m2/m2",
            "tool_area_m2",
            "1",
            "kalıp alanını bir kat kaplar",
        ),
        Suggestion(
            "BREATHER",
            "Breather",
            1.0,
            "m2/m2",
            "tool_area_m2",
            "1",
            "kalıp alanını bir kat kaplar",
        ),
        Suggestion(
            "VAKUM_TORBA",
            "Vakum torba",
            1.0 + i.bag_extra,
            "m2/m2",
            "tool_area_m2",
            f"1 × (1 + {i.bag_extra:g})",
            "torba kalıp alanını kaplar; pile/bindirme payı sizin girdiğiniz",
        ),
        Suggestion(
            "SIZDIRMAZLIK_BANDI",
            "Sızdırmazlık bandı",
            float(i.tape_rows),
            "m/m",
            "tool_perimeter_m",
            f"{i.tape_rows} sıra",
            "kalıp (MEOP) çevresi boyunca; sıra sayısı sizin girdiğiniz",
        ),
    ]


SUGGESTED = frozenset(x.material for x in suggest(ConsumableInputs(skins=1)))
