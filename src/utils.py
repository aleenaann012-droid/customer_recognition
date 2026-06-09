def get_age_group(age):

    if age <= 17:
        return "0-17"

    elif age <= 25:
        return "18-25"

    elif age <= 35:
        return "26-35"

    elif age <= 45:
        return "36-45"

    elif age <= 60:
        return "46-60"

    else:
        return "60+"