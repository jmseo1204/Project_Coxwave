from enum import Enum

class Command(Enum):
    CONTINUE = 1  # 계속 진행 상태
    END = 0       # 종료 커맨드

def get_command(user_input):
    normalized_input = user_input.strip()
    
    if '끝' in normalized_input or '그만' in normalized_input:
        return Command.END
    else:
        return Command.CONTINUE