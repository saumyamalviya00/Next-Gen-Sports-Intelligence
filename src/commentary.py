class CommentaryGenerator:
    def __init__(self, use_local=False):
        self.use_local = use_local
        self.local_phrases = {
            "left wing cross": "Player {player} crosses from the left flank!",
            "right wing cross": "Player {player} delivers a cross from the right!",
            "default": "Action by player {player}"
        }

    def generate(self, action_type, player_id, **kwargs):
        """Robust commentary generation with fallback"""
        
        try:
            player_ref = f"ID:{player_id}"
            if 'jersey_num' in kwargs and kwargs['jersey_num']:
                player_ref = f"#{kwargs['jersey_num']}"
                
            if self.use_local or not hasattr(self, 'client'):
                return self.local_phrases.get(action_type, self.local_phrases["default"]).format(
                    player=player_ref,
                    **kwargs
                )
            # (Keep existing API logic if using external service)
        except Exception as e:
            return f"Action by {player_ref}"  # Fallback message