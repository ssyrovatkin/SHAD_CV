from rare_traffic_sign_solution import generate_one_icon, generate_all_data


if __name__ == '__main__':

    args = [
        './icons/1.1.png',
        './synt-imgs/',
        './background_images/',
        10
    ]

    generate_one_icon(args)

    # generate_all_data('./cropped-train/', './icons/', './background_images/', samples_per_class=1000)
