FROM nginx:alpine

# copy static files
COPY . /usr/share/nginx/html

# copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

