from typing import Dict, List


HERO_PROFILES = {
    "Lux": {
        "type": "mage",
        "skills": {
            "q": {"name": "light_binding", "type": "skillshot", "range": 1175},
            "w": {"name": "prismatic_barrier", "type": "shield", "range": 0},
            "e": {"name": "lucent_singularity", "type": "aoe", "range": 1100},
            "r": {"name": "final_spark", "type": "ultimate", "range": 3340}
        },
        "playstyle": {
            "preferred_range": 550,
            "aggressiveness": 0.7,
            "team_fight_role": "damage",
            "primary_focus": "poke"
        }
    },
    
    "Ahri": {
        "type": "mage",
        "skills": {
            "q": {"name": "orb_of_deception", "type": "skillshot", "range": 880},
            "w": {"name": "fox_fire", "type": "aoe", "range": 600},
            "e": {"name": "charm", "type": "skillshot", "range": 975},
            "r": {"name": "spirit_rush", "type": "dash", "range": 450}
        },
        "playstyle": {
            "preferred_range": 550,
            "aggressiveness": 0.8,
            "team_fight_role": "damage",
            "primary_focus": "burst"
        }
    },
    
    "Jinx": {
        "type": "marksman",
        "skills": {
            "q": {"name": "switcheroo", "type": "toggle", "range": 0},
            "w": {"name": "zap", "type": "skillshot", "range": 1450},
            "e": {"name": "flame_chompers", "type": "trap", "range": 925},
            "r": {"name": "super_mega_death_rocket", "type": "ultimate", "range": 20000}
        },
        "playstyle": {
            "preferred_range": 600,
            "aggressiveness": 0.6,
            "team_fight_role": "damage",
            "primary_focus": "sustained"
        }
    },
    
    "Ezreal": {
        "type": "marksman",
        "skills": {
            "q": {"name": "mystic_shot", "type": "skillshot", "range": 1150},
            "w": {"name": "essence_flux", "type": "skillshot", "range": 1150},
            "e": {"name": "arcane_shift", "type": "dash", "range": 475},
            "r": {"name": "trueshot_barrage", "type": "ultimate", "range": 20000}
        },
        "playstyle": {
            "preferred_range": 550,
            "aggressiveness": 0.5,
            "team_fight_role": "damage",
            "primary_focus": "poke"
        }
    },
    
    "Yasuo": {
        "type": "assassin",
        "skills": {
            "q": {"name": "steel_tempest", "type": "skillshot", "range": 475},
            "w": {"name": "wind_wall", "type": "wall", "range": 400},
            "e": {"name": "sweeping_blade", "type": "dash", "range": 475},
            "r": {"name": "last_breath", "type": "ultimate", "range": 1400}
        },
        "playstyle": {
            "preferred_range": 300,
            "aggressiveness": 0.9,
            "team_fight_role": "damage",
            "primary_focus": "burst"
        }
    },
    
    "Zed": {
        "type": "assassin",
        "skills": {
            "q": {"name": "razor_shuriken", "type": "skillshot", "range": 900},
            "w": {"name": "living_shadow", "type": "placement", "range": 650},
            "e": {"name": "shadow_slash", "type": "aoe", "range": 290},
            "r": {"name": "death_mark", "type": "ultimate", "range": 625}
        },
        "playstyle": {
            "preferred_range": 300,
            "aggressiveness": 0.85,
            "team_fight_role": "assassin",
            "primary_focus": "burst"
        }
    },
    
    "Lee Sin": {
        "type": "fighter",
        "skills": {
            "q": {"name": "sonic_wave", "type": "skillshot", "range": 1100},
            "w": {"name": "safeguard", "type": "shield_dash", "range": 700},
            "e": {"name": "tempest", "type": "aoe", "range": 350},
            "r": {"name": "dragon_rage", "type": "ultimate", "range": 375}
        },
        "playstyle": {
            "preferred_range": 250,
            "aggressiveness": 0.8,
            "team_fight_role": "damage",
            "primary_focus": "burst"
        }
    },
    
    "Thresh": {
        "type": "support",
        "skills": {
            "q": {"name": "death_sentence", "type": "skillshot", "range": 1075},
            "w": {"name": "dark_passage", "type": "dash", "range": 950},
            "e": {"name": "flay", "type": "skillshot", "range": 500},
            "r": {"name": "the_box", "type": "ultimate", "range": 450}
        },
        "playstyle": {
            "preferred_range": 400,
            "aggressiveness": 0.4,
            "team_fight_role": "utility",
            "primary_focus": "control"
        }
    },
    
    "Blitzcrank": {
        "type": "support",
        "skills": {
            "q": {"name": "rocket_grab", "type": "skillshot", "range": 1150},
            "w": {"name": "overdrive", "type": "buff", "range": 0},
            "e": {"name": "power_fist", "type": "melee", "range": 0},
            "r": {"name": "static_field", "type": "aoe", "range": 600}
        },
        "playstyle": {
            "preferred_range": 350,
            "aggressiveness": 0.7,
            "team_fight_role": "utility",
            "primary_focus": "control"
        }
    },
    
    "Morgana": {
        "type": "support",
        "skills": {
            "q": {"name": "dark_binding", "type": "skillshot", "range": 1250},
            "w": {"name": "tormented_soil", "type": "aoe", "range": 900},
            "e": {"name": "black_shield", "type": "shield", "range": 800},
            "r": {"name": "soul_shackles", "type": "ultimate", "range": 600}
        },
        "playstyle": {
            "preferred_range": 450,
            "aggressiveness": 0.5,
            "team_fight_role": "utility",
            "primary_focus": "control"
        }
    },
    
    "Veigar": {
        "type": "mage",
        "skills": {
            "q": {"name": "baleful_strike", "type": "skillshot", "range": 900},
            "w": {"name": "dark_matter", "type": "aoe", "range": 950},
            "e": {"name": "event_horizon", "type": "zone", "range": 725},
            "r": {"name": "primordial_burst", "type": "ultimate", "range": 650}
        },
        "playstyle": {
            "preferred_range": 500,
            "aggressiveness": 0.6,
            "team_fight_role": "damage",
            "primary_focus": "burst"
        }
    },
    
    "Teemo": {
        "type": "marksman",
        "skills": {
            "q": {"name": "blinding_dart", "type": "skillshot", "range": 680},
            "w": {"name": "move_quick", "type": "buff", "range": 0},
            "e": {"name": "toxic_shot", "type": "passive", "range": 0},
            "r": {"name": "noxious_trap", "type": "trap", "range": 400}
        },
        "playstyle": {
            "preferred_range": 550,
            "aggressiveness": 0.4,
            "team_fight_role": "damage",
            "primary_focus": "poke"
        }
    },
    
    "Garen": {
        "type": "tank",
        "skills": {
            "q": {"name": "decisive_strike", "type": "buff", "range": 0},
            "w": {"name": "courage", "type": "buff", "range": 0},
            "e": {"name": "judgment", "type": "aoe", "range": 325},
            "r": {"name": "demacian_justice", "type": "ultimate", "range": 400}
        },
        "playstyle": {
            "preferred_range": 200,
            "aggressiveness": 0.75,
            "team_fight_role": "tank",
            "primary_focus": "damage"
        }
    },
    
    "Darius": {
        "type": "fighter",
        "skills": {
            "q": {"name": "decimate", "type": "aoe", "range": 425},
            "w": {"name": "crippling_strike", "type": "melee", "range": 0},
            "e": {"name": "apprehend", "type": "skillshot", "range": 550},
            "r": {"name": "noxian_guillotine", "type": "ultimate", "range": 460}
        },
        "playstyle": {
            "preferred_range": 200,
            "aggressiveness": 0.85,
            "team_fight_role": "damage",
            "primary_focus": "sustained"
        }
    },
    
    "Miss Fortune": {
        "type": "marksman",
        "skills": {
            "q": {"name": "double_up", "type": "skillshot", "range": 650},
            "w": {"name": "love_tap", "type": "passive", "range": 0},
            "e": {"name": "strut", "type": "buff", "range": 0},
            "r": {"name": "bullet_time", "type": "ultimate", "range": 1400}
        },
        "playstyle": {
            "preferred_range": 600,
            "aggressiveness": 0.65,
            "team_fight_role": "damage",
            "primary_focus": "sustained"
        }
    }
}


def get_hero_profile(hero_name: str) -> Dict:
    return HERO_PROFILES.get(hero_name, {})


def get_heroes_by_type(hero_type: str) -> List[str]:
    return [name for name, profile in HERO_PROFILES.items() 
            if profile.get("type") == hero_type]
